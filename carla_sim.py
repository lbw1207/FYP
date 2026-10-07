#!/usr/bin/env python
"""
Perceptinet - CARLA runner (written against the CARLA 0.9.14-0.9.16 Python API).

Blind-spot scenario: an ego vehicle drives down a straight road, a parked truck hides a pedestrian
standing on the kerb behind it. An overhead RSU camera on the opposite side of the road sees the
pedestrian; the ego camera cannot. Both feeds go through YOLO, detections are projected onto the
ground plane (shared world grid), the RSU sends only class + coordinates over the V2I link, the
vehicle fuses them, and the decision (CLEAR / SLOW / STOP) is executed in CARLA and shown live
on the dashboard.

    1. start the CARLA server            (./CarlaUE4.sh   or   CarlaUE4.exe)
    2. python carla_sim.py --weights yolov8n.pt          # or your CARLA-trained best.pt
    3. open http://localhost:8000
"""
import argparse
import math
import queue
import random
import sys
import time
from collections import deque

import cv2
import numpy as np

try:
    import carla
except ImportError:
    sys.exit("The 'carla' Python package is not installed:  pip install carla==0.9.15  (match your server version)")

from dashboard_server import DashboardServer
from detector import Detector, draw, to_b64_jpeg
from fusion import group_of, pixel_to_ground, to_ego_frame
from pipeline import Pipeline
from transport import make_transport

# ------------------------------------------------------------------ scenario geometry
TRUCK_DIST = 42.0           # truck centre, metres ahead of the ego spawn
PED_LATERAL = 3.1           # pedestrian offset to the right of the lane centre (on the kerb / verge)
TRIGGER_DIST = 22.0         # pedestrian starts crossing when the ego is this close [m]
WALK_SPEED, WALK_TIME = 1.5, 6.5
RSU_SIDE_OFFSET, RSU_HEIGHT, RSU_PITCH = 6.5, 8.0, -35.0     # RSU pole on the LEFT of the road
EGO_CAM = carla.Transform(carla.Location(x=1.5, z=1.6))


class SensorQueue:
    """Holds camera frames so we can grab the one that matches the current tick (synchronous mode)."""
    def __init__(self):
        self.q = queue.Queue()

    def __call__(self, image):
        self.q.put(image)

    def get(self, frame, timeout=5.0):
        end = time.time() + timeout
        while time.time() < end:
            try:
                img = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if img.frame >= frame:
                return img
        raise RuntimeError("camera frame timeout")


def to_bgr(image):
    a = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return a[:, :, :3].copy()


def find_straight_spawn(cmap, need=95.0, index=-1):
    pts = cmap.get_spawn_points()
    order = [pts[index]] if index >= 0 else random.sample(pts, len(pts))
    for tr in order:
        wp, ok, dist = cmap.get_waypoint(tr.location), True, 0.0
        yaw0 = wp.transform.rotation.yaw
        while dist < need:
            nxt = wp.next(5.0)
            if not nxt or len(nxt) != 1 or nxt[0].is_junction:
                ok = False
                break
            wp, dist = nxt[0], dist + 5.0
            if abs((wp.transform.rotation.yaw - yaw0 + 180) % 360 - 180) > 4.0:
                ok = False
                break
        if ok:
            return tr
    raise RuntimeError("no straight road segment found - try --town Town05 or pass --spawn-index")


class Scenario:
    def __init__(self, world, tm, args):
        self.world, self.tm, self.args = world, tm, args
        self.cmap, self.actors = world.get_map(), []
        self.lib = world.get_blueprint_library()
        self.spawn_tf = find_straight_spawn(self.cmap, index=args.spawn_index)   # fixed for restarts
        self.build()

    # -------------------------------------------------------------- construction
    def _spawn(self, bp, tf, **kw):
        a = self.world.try_spawn_actor(bp, tf, **kw)
        if a is not None:
            self.actors.append(a)
        return a

    def _offset(self, tf, fwd=0.0, right=0.0, up=0.0, yaw=None):
        f, r = tf.get_forward_vector(), tf.get_right_vector()
        loc = carla.Location(tf.location.x + f.x * fwd + r.x * right, tf.location.y + f.y * fwd + r.y * right,
                             tf.location.z + up)
        return carla.Transform(loc, carla.Rotation(yaw=tf.rotation.yaw if yaw is None else yaw))

    def build(self):
        w, a = self.world, self.args
        road_tf = self.spawn_tf
        self.ground_z = self.cmap.get_waypoint(road_tf.location).transform.location.z
        self.road_yaw = road_tf.rotation.yaw
        self.t_start = None

        # ego + cameras + collision sensor -----------------------------------------
        ego_bp = self.lib.find("vehicle.tesla.model3")
        ego_bp.set_attribute("role_name", "ego")
        start = carla.Transform(carla.Location(road_tf.location.x, road_tf.location.y, road_tf.location.z + 0.3),
                                road_tf.rotation)
        self.ego = self._spawn(ego_bp, start)
        if self.ego is None:
            raise RuntimeError("could not spawn the ego vehicle")
        self.ego_q, self.rsu_q, self.collisions = SensorQueue(), SensorQueue(), deque()
        self.ego_cam = self._camera(EGO_CAM, self.ego, self.ego_q)
        col = self._spawn(self.lib.find("sensor.other.collision"), carla.Transform(), attach_to=self.ego)
        col.listen(lambda ev: self.collisions.append(ev.other_actor.type_id))

        # parked truck (the occluder) ----------------------------------------------
        truck_bp = None
        for name in ("vehicle.carlamotors.european_hgv", "vehicle.carlamotors.carlacola", "vehicle.carlamotors.firetruck",
                     "vehicle.mitsubishi.fusorosa"):
            found = self.lib.filter(name)
            if found:
                truck_bp = found[0]
                break
        self.truck, truck_tf = None, None
        for right in (2.0, 1.6, 1.2, 0.8):                    # step towards the road until the spawn succeeds
            truck_tf = self._offset(road_tf, fwd=TRUCK_DIST, right=right, up=0.4)
            self.truck = self._spawn(truck_bp, truck_tf)
            if self.truck:
                break
        if self.truck is None:
            raise RuntimeError("could not spawn the parked truck - try another --spawn-index")
        self.truck.set_simulate_physics(False)
        ext = self.truck.bounding_box.extent
        self.truck_info = {"x": truck_tf.location.x, "y": truck_tf.location.y, "yaw": self.road_yaw,
                           "length": 2 * ext.x, "width": 2 * ext.y}

        # pedestrian, hidden just behind the truck's rear end ----------------------
        ped_fwd = TRUCK_DIST + ext.x + 1.6
        ped_bp = self.lib.filter("walker.pedestrian.0001")[0]
        self.ped = None
        for dz in (0.6, 1.0, 1.5):
            self.ped = self._spawn(ped_bp, self._offset(road_tf, fwd=ped_fwd, right=PED_LATERAL, up=dz))
            if self.ped:
                break
        if self.ped is None:
            raise RuntimeError("could not spawn the pedestrian")
        self.ped_dir = road_tf.get_right_vector()             # crossing = towards the LEFT (-right)
        self.walking, self.walk_t = False, 0.0

        # RSU camera on the opposite side, looking across the road and behind the truck ---
        rsu_tf = self._offset(road_tf, fwd=TRUCK_DIST + ext.x * 0.6, right=-RSU_SIDE_OFFSET, up=RSU_HEIGHT,
                              yaw=self.road_yaw + 90.0)
        rsu_tf.rotation.pitch = RSU_PITCH
        self.rsu_cam = self._camera(rsu_tf, None, self.rsu_q)
        self.rsu_info = {"x": rsu_tf.location.x, "y": rsu_tf.location.y}

        # autopilot for the ego, our pipeline is the only thing allowed to react to pedestrians
        self.ego.set_autopilot(True, self.tm.get_port())
        self.tm.ignore_walkers_percentage(self.ego, 100.0)
        self.tm.auto_lane_change(self.ego, False)
        if hasattr(self.tm, "set_desired_speed"):
            self.tm.set_desired_speed(self.ego, a.target_speed)
        else:
            self.tm.vehicle_percentage_speed_difference(self.ego, 0.0)
        self.autopilot = True

    def _camera(self, tf, parent, q):
        bp = self.lib.find("sensor.camera.rgb")
        for k, v in (("image_size_x", self.args.width), ("image_size_y", self.args.height), ("fov", self.args.fov)):
            bp.set_attribute(k, str(v))
        cam = self.world.spawn_actor(bp, tf, attach_to=parent)
        cam.listen(q)
        self.actors.append(cam)
        return cam

    def destroy(self):
        for a in reversed(self.actors):
            try:
                if hasattr(a, "stop"):
                    a.stop()
                a.destroy()
            except RuntimeError:
                pass
        self.actors = []

    def rebuild(self):
        self.destroy()
        self.build()

    # -------------------------------------------------------------- per-tick logic
    def step(self, sim_t):
        """Pedestrian trigger + walking. Returns True if a new pedestrian collision happened."""
        if not self.walking and self.ped_dist() < TRIGGER_DIST:
            self.walking, self.walk_t = True, 0.0
        if self.walking:
            self.walk_t += self.world.get_settings().fixed_delta_seconds
            moving = self.walk_t < WALK_TIME
            d = self.ped_dir
            self.ped.apply_control(carla.WalkerControl(carla.Vector3D(-d.x, -d.y, 0.0), WALK_SPEED if moving else 0.0))
        hit = any("walker" in c for c in self.collisions)
        self.collisions.clear()
        return hit

    def ped_dist(self):
        return self.ego.get_location().distance(self.ped.get_location())

    def ego_state(self):
        tr, v = self.ego.get_transform(), self.ego.get_velocity()
        return {"x": tr.location.x, "y": tr.location.y, "yaw": tr.rotation.yaw,
                "speed": math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)}

    def apply_decision(self, state):
        """STOP = emergency brake (autopilot off). SLOW = throttle the autopilot. CLEAR = normal."""
        if state == "STOP":
            if self.autopilot:
                self.ego.set_autopilot(False, self.tm.get_port())
                self.autopilot = False
            self.ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
            return
        if not self.autopilot:
            self.ego.set_autopilot(True, self.tm.get_port())
            self.autopilot = True
        self.tm.vehicle_percentage_speed_difference(self.ego, 55.0 if state == "SLOW" else 0.0)
        if state == "CLEAR" and hasattr(self.tm, "set_desired_speed"):
            self.tm.set_desired_speed(self.ego, self.args.target_speed)


def run_detection(detector, image, ground_z, ego_z=None, exclude_near=None):
    """YOLO on one camera frame -> (frame, kept-detections-with-boxes, labels, world-space detections)."""
    frame = to_bgr(image)
    cam_m = np.array(image.transform.get_matrix())
    dets = detector.detect(frame)
    kept, labels, out = [], [], []
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        pt = pixel_to_ground((x1 + x2) / 2.0, y2, cam_m, image.width, image.height, image.fov, ground_z)
        if pt is None:
            continue
        if exclude_near is not None and group_of(d["cls"]) == "vehicle" and \
                math.hypot(pt[0] - exclude_near[0], pt[1] - exclude_near[1]) < 4.5:
            continue                                            # the RSU also sees the ego car - drop it
        out.append({"cls": d["cls"], "conf": d["conf"], "x": pt[0], "y": pt[1]})
        kept.append(d)
        labels.append(None)
    return frame, kept, labels, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town03", help="'current' keeps whatever map is loaded")
    ap.add_argument("--weights", default="yolov8n.pt", help="use your CARLA-trained best.pt here")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="e.g. 0 for the first GPU, cpu")
    ap.add_argument("--classes", nargs="*", default=None, help="class names to keep (default: COCO road users)")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=450)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--fps", type=int, default=20, help="simulation ticks per simulated second")
    ap.add_argument("--detect-every", type=int, default=2, help="run YOLO every N ticks")
    ap.add_argument("--target-speed", type=float, default=35.0, help="km/h")
    ap.add_argument("--spawn-index", type=int, default=-1)
    ap.add_argument("--dash-port", type=int, default=8000)
    ap.add_argument("--transport", choices=["udp", "mqtt"], default="udp")
    ap.add_argument("--mqtt-host", default="127.0.0.1")
    a = ap.parse_args()

    detector = Detector(a.weights, a.conf, a.imgsz, a.device, a.classes)
    pipe = Pipeline(make_transport(a.transport, a.mqtt_host))
    flags = {"reset": False}

    def on_command(c):
        if c["cmd"] == "set_coop":
            pipe.coop = bool(c["value"])
        elif c["cmd"] == "set_latency":
            pipe.latency_ms = float(c["value"])
        elif c["cmd"] == "reset":
            flags["reset"] = True

    dash = DashboardServer(a.dash_port, on_command)
    dash.start()

    client = carla.Client(a.host, a.port)
    client.set_timeout(30.0)
    world = client.get_world() if a.town == "current" else client.load_world(a.town)
    orig = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 1.0 / a.fps
    world.apply_settings(settings)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    world.tick()

    scn = Scenario(world, tm, a)
    frames, last_push, tick, fps, t_prev = {}, 0.0, 0, 0.0, time.time()
    onboard, rsu_dets = [], []
    try:
        while True:
            if flags["reset"]:
                pipe.reset()
                scn.rebuild()
                flags["reset"], onboard, rsu_dets, frames = False, [], [], {}
            frame_id = world.tick()
            sim_t = world.get_snapshot().timestamp.elapsed_seconds
            img_e, img_r = scn.ego_q.get(frame_id), scn.rsu_q.get(frame_id)
            ego = scn.ego_state()

            if scn.step(sim_t):
                pipe.collision(sim_t)

            if tick % a.detect_every == 0:
                fe, ke, le, onboard = run_detection(detector, img_e, scn.ground_z)
                fr, kr, lr, rsu_dets = run_detection(detector, img_r, scn.ground_z, exclude_near=(ego["x"], ego["y"]))
                # distance labels on the vehicle view
                dl = [f'{math.hypot(o["x"] - ego["x"], o["y"] - ego["y"]):.0f} m' for o in onboard]
                b64_v, _ = to_b64_jpeg(draw(fe, ke, dl))
                b64_r, rsu_bytes = to_b64_jpeg(draw(fr, kr, lr))
                frames = {"vehicle": b64_v, "rsu": b64_r}
                pipe.rsu_publish(sim_t, rsu_dets, rsu_bytes)

            time.sleep(0.001)                                       # give the UDP/MQTT thread a moment
            res = pipe.vehicle_step(sim_t, ego, onboard)
            scn.apply_decision(res["decision"]["state"])

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now
            if now - last_push > 0.1:
                dash.push(pipe.make_state("carla", sim_t, fps, ego, res,
                                          {"truck": scn.truck_info, "rsu": scn.rsu_info}, frames))
                last_push = now
            tick += 1
    except KeyboardInterrupt:
        print("\n[carla_sim] stopping ...")
    finally:
        try:
            scn.destroy()
            tm.set_synchronous_mode(False)
            world.apply_settings(orig)
        finally:
            dash.stop()
            pipe.transport.close()


if __name__ == "__main__":
    main()
