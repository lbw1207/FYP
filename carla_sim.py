#!/usr/bin/env python
"""
Perceptinet - CARLA runner (written against the CARLA 0.9.14-0.9.16 Python API).

Two blind-spot scenarios (--scenario):
  intersection  (default) the ego drives straight towards a junction. A delivery truck (simulated, physics
                off) is parked half on the kerb just before the crossing, and a pedestrian waiting in front
                of it steps out across the road. An overhead RSU on the far corner of the junction sees
                the pedestrian; the ego camera cannot.
  parked_truck  the original straight-road version: a parked truck hides a pedestrian on the kerb.
Both feeds go through YOLO, detections are projected onto the ground plane (shared world grid), the
RSU sends only class + coordinates over the V2I link, the vehicle fuses them, and the decision
(CLEAR / SLOW / STOP) is executed in CARLA and shown live on the dashboard.

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

import numpy as np

try:
    import carla
except ImportError:
    sys.exit("The 'carla' Python package is not installed:  pip install carla==0.9.15  (match your server version)")

from dashboard_server import DashboardServer
from detector import Detector, to_b64_jpeg
from fusion import group_of, pixel_to_ground, to_ego_frame
from pipeline import Pipeline
from transport import make_transport

# ------------------------------------------------------------------ scenario geometry
# parked_truck scenario
TRUCK_DIST = 42.0           # truck centre, metres ahead of the ego spawn
PED_LATERAL = 3.1           # pedestrian offset to the right of the lane centre (on the kerb / verge)
TRIGGER_DIST = 22.0         # pedestrian starts crossing when the ego is this close [m]
WALK_SPEED, WALK_TIME = 1.5, 6.5
RSU_SIDE_OFFSET, RSU_HEIGHT, RSU_PITCH = 6.5, 8.0, -35.0     # RSU pole on the LEFT of the road
# intersection scenario (distances along the ego road, relative to where the junction starts)
JX_MIN_DIST, JX_MAX_DIST = 45.0, 90.0   # how far ahead of the ego spawn the junction may be
JX_CROSSING_BACK = 1.5      # truck front / crossing this far before the junction entry [m]
JX_TRUCK_KERB = 1.4         # truck centre this far right of the road's right edge (half on the kerb)
JX_PED_GAPS = (0.6, 0.9, 1.2)   # pedestrian this far in front of the truck's nose (first that spawns) [m]
JX_PED_SPEED = 2.5          # the pedestrian steps out briskly [m/s] - a slow walk gives the onboard camera time
JX_TRIGGER_DIST = None      # step out when the ego is this far before the pedestrian (along the road) [m];
                            # None = derived so the pedestrian reaches the ego lane exactly when a car still
                            # driving at --target-speed arrives, i.e. too late for the onboard camera alone
JX_TRIGGER_EXTRA = 2.3      # ego location is the car's centre; its front bumper is ~2.3 m further forward
EGO_CAM = carla.Transform(carla.Location(x=1.5, z=1.6))
SLOW_FACTOR = 0.4           # SLOW = drive at 40 % of --target-speed (same as the mock)
SCENARIO_NAMES = {"intersection": "Blind intersection - pedestrian on crossing hidden by parked delivery truck",
                  "parked_truck": "Blind spot - pedestrian behind parked truck"}
TRUCK_BPS = ("vehicle.carlamotors.carlacola", "vehicle.carlamotors.european_hgv", "vehicle.carlamotors.firetruck",
             "vehicle.mitsubishi.fusorosa")


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


def find_junction_spawn(cmap, index=-1, lo=JX_MIN_DIST, hi=JX_MAX_DIST):
    """A spawn point on a straight road that reaches a junction after lo..hi metres.
    Returns (spawn transform, distance to the junction entry, last waypoint before the junction, junction)."""
    pts = cmap.get_spawn_points()
    order = [pts[index]] if index >= 0 else random.sample(pts, len(pts))
    for tr in order:
        wp, dist = cmap.get_waypoint(tr.location), 0.0
        yaw0 = wp.transform.rotation.yaw
        if wp.is_junction:
            continue
        while dist <= hi:
            nxt = wp.next(2.0)
            if not nxt or abs((nxt[0].transform.rotation.yaw - yaw0 + 180) % 360 - 180) > 4.0:
                break
            if nxt[0].is_junction:
                if dist >= lo:
                    return tr, dist, wp, nxt[0].get_junction()
                break
            if len(nxt) != 1:
                break
            wp, dist = nxt[0], dist + 2.0
    raise RuntimeError("no straight road leading into a junction found - try another --town or --spawn-index")


def road_edges(wp):
    """Distance from the lane centre to the right / left edge of the drivable road [m]."""
    def walk(first_step):
        edge, w, seen = wp.lane_width / 2.0, wp, {wp.lane_id}
        for _ in range(8):
            # after crossing the centre line the opposite lanes' "right" points further away from us
            nxt = first_step(w) if w.lane_id * wp.lane_id > 0 else w.get_right_lane()
            if nxt is None or nxt.lane_type != carla.LaneType.Driving or nxt.lane_id in seen:
                break
            seen.add(nxt.lane_id)
            edge, w = edge + nxt.lane_width, nxt
        return edge
    return walk(lambda w: w.get_right_lane()), walk(lambda w: w.get_left_lane())


class Scenario:
    def __init__(self, world, tm, args):
        self.world, self.tm, self.args = world, tm, args
        self.cmap, self.actors = world.get_map(), []
        self.lib = world.get_blueprint_library()
        if args.scenario == "intersection":                                       # fixed for restarts
            self.spawn_tf, self.jx_dist, self.jx_wp, self.junction = find_junction_spawn(self.cmap, args.spawn_index)
        else:
            self.spawn_tf = find_straight_spawn(self.cmap, index=args.spawn_index)
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

    def _ground_z(self, loc):
        """Height of whatever is under `loc` (road or kerb); falls back to the road height."""
        if hasattr(self.world, "ground_projection"):          # CARLA >= 0.9.12: ray-cast straight down
            hit = self.world.ground_projection(carla.Location(loc.x, loc.y, loc.z + 5.0), 15.0)
            if hit is not None:
                return hit.location.z
        return self.cmap.get_waypoint(loc).transform.location.z

    def _truck(self, fwd, rights):
        """Parked truck (the simulated occluder): physics off so it stays exactly where we put it."""
        bp = next((found[0] for found in (self.lib.filter(n) for n in TRUCK_BPS) if found), None)
        for right in rights:                                  # step until the spawn succeeds
            tf = self._offset(self.spawn_tf, fwd=fwd, right=right, up=0.4)
            ground = self._ground_z(tf.location)              # measure before spawning (the ray would hit the truck)
            truck = self._spawn(bp, tf)
            if truck:
                # spawning needs clearance above the road, and with physics off the truck never falls,
                # so lower it until the bottom of its bounding box (the wheels) touches the ground
                truck.set_simulate_physics(False)
                bb = truck.bounding_box
                tf.location.z = ground - (bb.location.z - bb.extent.z)
                truck.set_transform(tf)
                return truck, tf, bb.extent
        raise RuntimeError("could not spawn the parked truck - try another --spawn-index")

    def _pedestrian(self, fwd, right):
        ped = self._try_pedestrian(fwd, right)
        if ped is None:
            raise RuntimeError("could not spawn the pedestrian")
        return ped

    def _try_pedestrian(self, fwd, right):
        bp = self.lib.filter("walker.pedestrian.0001")[0]
        for dz in (0.6, 1.0, 1.5):
            ped = self._spawn(bp, self._offset(self.spawn_tf, fwd=fwd, right=right, up=dz))
            if ped:
                return ped
        return None

    def _road_coords(self, loc):
        """World location -> (forward, right) metres in the ego road frame (relative to the ego spawn)."""
        f, r, o = self.spawn_tf.get_forward_vector(), self.spawn_tf.get_right_vector(), self.spawn_tf.location
        dx, dy = loc.x - o.x, loc.y - o.y
        return dx * f.x + dy * f.y, dx * r.x + dy * r.y

    def _rsu(self, fwd, right, look_at=None):
        tf = self._offset(self.spawn_tf, fwd=fwd, right=right, up=RSU_HEIGHT, yaw=self.road_yaw + 90.0)
        if look_at is not None:                               # aim the RSU at the blind spot
            tf.rotation.yaw = math.degrees(math.atan2(look_at.y - tf.location.y, look_at.x - tf.location.x))
        tf.rotation.pitch = RSU_PITCH
        self.rsu_cam = self._camera(tf, None, self.rsu_q)
        self.rsu_info = {"x": tf.location.x, "y": tf.location.y}

    def build(self):
        a, road_tf = self.args, self.spawn_tf
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

        self.junction_info = self.crossing_info = None
        if a.scenario == "intersection":
            self._build_intersection()
        else:
            self._build_parked_truck()
        self.ped_dir = road_tf.get_right_vector()             # crossing = towards the LEFT (-right)
        self.walking, self.walk_t = False, 0.0

        # autopilot for the ego, our pipeline is the only thing allowed to react to pedestrians
        self.ego.set_autopilot(True, self.tm.get_port())
        # The Traffic Manager must not brake on its own, or the car slows down even with cooperative
        # perception OFF: it ignores walkers, lights, signs and the parked truck (which sits at the lane edge).
        self.tm.ignore_walkers_percentage(self.ego, 100.0)
        self.tm.ignore_vehicles_percentage(self.ego, 100.0)
        self.tm.ignore_lights_percentage(self.ego, 100.0)
        self.tm.ignore_signs_percentage(self.ego, 100.0)
        self.tm.auto_lane_change(self.ego, False)
        if hasattr(self.tm, "set_route"):
            self.tm.set_route(self.ego, ["Straight"] * 3)     # go straight through the junction
        if hasattr(self.tm, "set_desired_speed"):
            self.tm.set_desired_speed(self.ego, a.target_speed)
        else:
            self.tm.vehicle_percentage_speed_difference(self.ego, 0.0)
        self.autopilot = True
        self.reached_speed, self.slow_since, self.tm_slow_logged = False, None, False

    def _build_parked_truck(self):
        self.truck, truck_tf, ext = self._truck(TRUCK_DIST, (2.0, 1.6, 1.2, 0.8))
        self.blocker_info = {"x": truck_tf.location.x, "y": truck_tf.location.y, "yaw": self.road_yaw,
                             "length": 2 * ext.x, "width": 2 * ext.y, "height": 2 * ext.z, "label": "parked truck"}
        # pedestrian, hidden just behind the truck's rear end
        self.ped = self._pedestrian(TRUCK_DIST + ext.x + 1.6, PED_LATERAL)
        self.trigger, self.walk_time, self.walk_speed = TRIGGER_DIST, WALK_TIME, WALK_SPEED
        # RSU camera on the opposite side, looking across the road and behind the truck
        self._rsu(TRUCK_DIST + ext.x * 0.6, -RSU_SIDE_OFFSET)

    def _build_intersection(self):
        right_edge, left_edge = road_edges(self.jx_wp)
        # delivery truck parked half on the kerb, its nose just short of the crossing (truck sizes differ
        # between blueprints, so everything below is placed from the extent of the truck that actually spawned)
        nose = self.jx_dist - JX_CROSSING_BACK
        self.truck, truck_tf, ext = self._truck(nose - 4.5, [right_edge + JX_TRUCK_KERB - d for d in (0.0, 0.4, 0.8, 1.2, 1.6)])
        bb = self.truck.bounding_box
        t_fwd, t_right = self._road_coords(truck_tf.location)
        t_fwd += bb.location.x                                # bounding-box centre (may be offset from the origin)
        self.blocker_info = {"x": truck_tf.location.x, "y": truck_tf.location.y, "yaw": self.road_yaw,
                             "length": 2 * ext.x, "width": 2 * ext.y, "height": 2 * ext.z, "label": "delivery truck (simulated)"}
        # pedestrian right in front of the truck's nose and centred on its width: the whole truck is then
        # between the ego camera and the pedestrian, so the onboard camera cannot see them until they step out
        self.ped, ped_fwd = None, None
        for gap in JX_PED_GAPS:
            ped_fwd = t_fwd + ext.x + gap
            self.ped = self._try_pedestrian(ped_fwd, t_right)
            if self.ped:
                break
        if self.ped is None:
            raise RuntimeError("could not spawn the pedestrian in front of the truck - try another --spawn-index")
        cross_fwd = ped_fwd
        self.walk_speed = JX_PED_SPEED
        self.walk_time = (t_right + left_edge + 2.0) / self.walk_speed
        # Step out so the pedestrian reaches the ego lane (|right| < 1 m) just as a full-speed car arrives.
        # The trigger is a fixed distance, so a car the RSU has already slowed down reaches it later and slower
        # and can stop; a car relying on its own camera only sees the pedestrian with a few metres to go.
        self.trigger = JX_TRIGGER_DIST or (self.args.target_speed / 3.6 * max(t_right - 1.0, 0.5) / self.walk_speed
                                           + JX_TRIGGER_EXTRA)
        print(f"[carla_sim] truck {2 * ext.x:.1f} x {2 * ext.y:.1f} m at {t_right:.1f} m right; "
              f"pedestrian {ped_fwd - t_fwd - ext.x:.1f} m in front of it; trigger {self.trigger:.1f} m")
        # junction footprint along the ego road -> RSU on the far-left corner, aimed at the pedestrian
        jb, fv = self.junction.bounding_box, self.spawn_tf.get_forward_vector()
        jlen = 2 * (abs(fv.x) * jb.extent.x + abs(fv.y) * jb.extent.y)
        self.world.tick()                                     # let the pedestrian settle before aiming at it
        self._rsu(self.jx_dist + jlen + 2.0, -(left_edge + 3.0), look_at=self.ped.get_location())
        self.junction_info = {"x": jb.location.x, "y": jb.location.y, "yaw": self.road_yaw + 90.0, "width": jlen}
        c = self._offset(self.spawn_tf, fwd=cross_fwd).location
        self.crossing_info = {"x": c.x, "y": c.y, "width": 3.0}

    def scenario_state(self):
        return {"name": SCENARIO_NAMES[self.args.scenario], "blocker": self.blocker_info, "rsu": self.rsu_info,
                "junction": self.junction_info, "crossing": self.crossing_info}

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
        if not self.walking and self.ped_ahead() < self.trigger:
            self.walking, self.walk_t = True, 0.0
        if self.walking:
            self.walk_t += self.world.get_settings().fixed_delta_seconds
            moving = self.walk_t < self.walk_time
            d = self.ped_dir
            self.ped.apply_control(carla.WalkerControl(carla.Vector3D(-d.x, -d.y, 0.0), self.walk_speed if moving else 0.0))
        hit = any("walker" in c for c in self.collisions)
        self.collisions.clear()
        return hit

    def ped_ahead(self):
        """How far the pedestrian is ahead of the ego, measured along the road [m]."""
        return self._road_coords(self.ped.get_location())[0] - self._road_coords(self.ego.get_location())[0]


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
        factor = SLOW_FACTOR if state == "SLOW" else 1.0
        if hasattr(self.tm, "set_desired_speed"):             # a desired speed overrides the percentage below
            self.tm.set_desired_speed(self.ego, self.args.target_speed * factor)
        else:
            self.tm.vehicle_percentage_speed_difference(self.ego, (1.0 - factor) * 100.0)

    def autopilot_slowdown(self, state, speed):
        """True (once per run) if the car is clearly slower than the target while Perceptinet says CLEAR,
        i.e. CARLA's own autopilot is braking - so a slowdown is never wrongly credited to the RSU."""
        target = self.args.target_speed / 3.6
        self.reached_speed = self.reached_speed or speed > 0.9 * target
        if not (self.reached_speed and state == "CLEAR" and speed < 0.7 * target):
            self.slow_since = None
            return False
        self.slow_since = self.slow_since or time.time()
        if not self.tm_slow_logged and time.time() - self.slow_since > 1.0:
            self.tm_slow_logged = True
            return True
        return False


def run_detection(detector, image, ground_z, exclude_near=None):
    """YOLO on one camera frame -> (raw frame, normalised boxes for the dashboard, world-space detections)."""
    frame = to_bgr(image)
    cam_m = np.array(image.transform.get_matrix())
    dets = detector.detect(frame)
    boxes, out = [], []
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        pt = pixel_to_ground((x1 + x2) / 2.0, y2, cam_m, image.width, image.height, image.fov, ground_z)
        if pt is None:
            continue
        if exclude_near is not None and group_of(d["cls"]) == "vehicle" and \
                math.hypot(pt[0] - exclude_near[0], pt[1] - exclude_near[1]) < 4.5:
            continue                                            # the RSU also sees the ego car - drop it
        out.append({"cls": d["cls"], "conf": d["conf"], "x": pt[0], "y": pt[1]})
        boxes.append({"cls": d["cls"], "conf": round(d["conf"], 2), "x": round(pt[0], 2), "y": round(pt[1], 2),
                      "box": [round(min(1.0, max(0.0, v)), 4) for v in
                              (x1 / image.width, y1 / image.height, x2 / image.width, y2 / image.height)]})
    return frame, boxes, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=["intersection", "parked_truck"], default="intersection")
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
    frames, boxes, ego_cam, last_push, tick, fps, t_prev = {}, {}, None, 0.0, 0, 0.0, time.time()
    onboard, rsu_dets = [], []
    try:
        while True:
            if flags["reset"]:
                pipe.reset()
                scn.rebuild()
                flags["reset"], onboard, rsu_dets, frames, boxes, ego_cam = False, [], [], {}, {}, None
            frame_id = world.tick()
            sim_t = world.get_snapshot().timestamp.elapsed_seconds
            img_e, img_r = scn.ego_q.get(frame_id), scn.rsu_q.get(frame_id)
            ego = scn.ego_state()

            if scn.step(sim_t):
                pipe.collision(sim_t)

            if tick % a.detect_every == 0:
                fe, box_e, onboard = run_detection(detector, img_e, scn.ground_z)
                fr, box_r, rsu_dets = run_detection(detector, img_r, scn.ground_z, exclude_near=(ego["x"], ego["y"]))
                b64_v, _ = to_b64_jpeg(fe)              # raw frames: the dashboard draws boxes (System mode only)
                b64_r, rsu_bytes = to_b64_jpeg(fr)
                frames, boxes = {"vehicle": b64_v, "rsu": b64_r}, {"vehicle": box_e, "rsu": box_r}
                ego_cam = {"matrix": np.array(img_e.transform.get_matrix()), "width": img_e.width,
                           "height": img_e.height, "fov": img_e.fov, "ground_z": scn.ground_z}
                pipe.rsu_publish(sim_t, rsu_dets, rsu_bytes)

            time.sleep(0.001)                                       # give the UDP/MQTT thread a moment
            res = pipe.vehicle_step(sim_t, ego, onboard)
            scn.apply_decision(res["decision"]["state"])
            if scn.autopilot_slowdown(res["decision"]["state"], ego["speed"]):
                pipe.note(sim_t, "warn", "Ego slowed by the CARLA autopilot, not by Perceptinet (decision is CLEAR)")

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now
            if now - last_push > 0.1:
                dash.push(pipe.make_state("carla", sim_t, fps, ego, res, scn.scenario_state(),
                                          frames, boxes, ego_cam))
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
