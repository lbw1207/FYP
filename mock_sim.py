#!/usr/bin/env python
"""
Perceptinet dashboard WITHOUT CARLA: a tiny kinematic replica of the blind-spot scenarios.
Uses exactly the same pipeline (transport -> late fusion -> decision) and the same dashboard,
so you can develop/demo the UI on any laptop and compare it with the CARLA run.

    python mock_sim.py                          ->  http://localhost:8000   (blind intersection)
    python mock_sim.py --scenario parked_truck  ->  pedestrian behind a parked truck
"""
import argparse
import math
import random
import time
from collections import deque

import cv2
import numpy as np

from dashboard_server import DashboardServer
from detector import to_b64_jpeg
from fusion import camera_matrix, world_to_pixel
from pipeline import Pipeline
from transport import make_transport

W, H, FOV, DT = 640, 360, 90, 0.05
ROAD_HALF = 5.25

# Both demo setups from the project description. Ego drives along +x on y = 0 (x forward, y right).
# "blocker" is the simulated occluder; cls = None means it is scenery (not a YOLO class).
SCENARIOS = {
    "intersection": {
        "name": "Blind intersection - pedestrian on zebra crossing hidden by parked delivery truck",
        "blocker": {"x": 39.5, "y": 3.75, "yaw": 0.0, "length": 9.0, "width": 2.5, "height": 3.6,
                    "label": "delivery truck (simulated box)", "cls": None},
        "junction": {"x": 53.0, "y": 0.0, "yaw": 90.0, "width": 10.5},
        "crossing": {"x": 46.5, "y": 0.0, "width": 2.5},
        "rsu": {"x": 59.0, "y": -8.5, "z": 8.0, "yaw": 134.0, "pitch": -35.0},    # far-left corner
        "ped": (46.5, 4.6), "ped_speed": 1.5, "trigger": 23.0, "cross_time": 8.0,
    },
    "parked_truck": {
        "name": "Blind spot - pedestrian behind parked truck",
        "blocker": {"x": 40.0, "y": 2.1, "yaw": 0.0, "length": 9.0, "width": 2.5, "height": 3.6,
                    "label": "parked truck", "cls": "truck"},
        "junction": None, "crossing": None,
        "rsu": {"x": 44.0, "y": -6.0, "z": 8.0, "yaw": 90.0, "pitch": -35.0},
        "ped": (48.5, 3.0), "ped_speed": 1.5, "trigger": 22.0, "cross_time": 6.0,
    },
}
SC = SCENARIOS["intersection"]


class World:
    def __init__(self, target_kmh):
        self.target = target_kmh / 3.6
        self.reset()

    def reset(self):
        self.t, self.ex, self.ev = 0.0, 0.0, 0.0
        self.px, self.py, self.walk_t = *SC["ped"], None
        self.hit = False
        self.hist = deque()                       # 0.4 s brake-actuation lag

    @property
    def ego(self):
        return {"x": self.ex, "y": 0.0, "yaw": 0.0, "speed": self.ev}

    def step(self, state):
        self.hist.append(state)
        state = self.hist.popleft() if len(self.hist) > 8 else "CLEAR"
        if state == "STOP":
            self.ev = max(0.0, self.ev - 6.0 * DT)
        else:
            tgt = self.target * (0.4 if state == "SLOW" else 1.0)
            self.ev += max(-4.0 * DT, min(2.5 * DT, tgt - self.ev))
        self.ex += self.ev * DT
        if self.walk_t is None and math.hypot(self.px - self.ex, self.py) < SC["trigger"]:
            self.walk_t = 0.0
        if self.walk_t is not None and self.walk_t < SC["cross_time"]:
            self.py -= SC["ped_speed"] * DT
            self.walk_t += DT
        self.t += DT
        newly_hit = (not self.hit) and abs(self.px - self.ex) < 2.5 and abs(self.py) < 1.2
        self.hit = self.hit or newly_hit
        return newly_hit

    # ---- what each camera can "detect" (stand-in for YOLO) -------------------
    def scene(self):
        """Everything that gets rendered (the blocker even when it is not a detectable class)."""
        b = SC["blocker"]
        return [{"cls": b["cls"], "x": b["x"], "y": b["y"], "h": b["height"], "l": b["length"], "w": b["width"], "id": "blocker"},
                {"cls": "person", "x": self.px, "y": self.py, "h": 1.75, "l": 0.5, "w": 0.5, "id": "ped"}]

    def objects(self):
        return [o for o in self.scene() if o["cls"]]

    def onboard_detect(self):
        out = []
        for o in self.objects():
            f = o["x"] - self.ex
            rng = 38.0 if o["cls"] == "person" else 70.0            # small objects are detected later
            if not (1 < f < rng) or abs(o["y"]) > f:                 # outside the 90-degree FOV
                continue
            if o["id"] != "blocker" and _occluded((self.ex + 1.5, 0.0), (o["x"], o["y"])):
                continue
            if random.random() < 0.05:
                continue
            out.append({"cls": o["cls"], "conf": random.uniform(.6, .9), "id": o["id"],
                        "x": o["x"] + random.gauss(0, .25), "y": o["y"] + random.gauss(0, .25)})
        return out

    def rsu_detect(self):
        out = []
        for o in self.objects():
            rsu = SC["rsu"]
            if o["id"] != "blocker" and _occluded((rsu["x"], rsu["y"]), (o["x"], o["y"])):
                continue
            if math.hypot(o["x"] - rsu["x"], o["y"] - rsu["y"]) < 45 and random.random() > 0.03:
                out.append({"cls": o["cls"], "conf": random.uniform(.65, .92), "id": o["id"],
                            "x": o["x"] + random.gauss(0, .25), "y": o["y"] + random.gauss(0, .25)})
        return out


def _occluded(p0, p1):
    """2-D segment vs. the blocker's footprint rectangle (slab test)."""
    b = SC["blocker"]
    xmin, xmax = b["x"] - b["length"] / 2, b["x"] + b["length"] / 2
    ymin, ymax = b["y"] - b["width"] / 2, b["y"] + b["width"] / 2
    dx, dy, t0, t1 = p1[0] - p0[0], p1[1] - p0[1], 0.0, 1.0
    for p, d, lo, hi in ((p0[0], dx, xmin, xmax), (p0[1], dy, ymin, ymax)):
        if abs(d) < 1e-9:
            if not lo <= p <= hi:
                return False
        else:
            a, b = sorted(((lo - p) / d, (hi - p) / d))
            t0, t1 = max(t0, a), min(t1, b)
            if t0 > t1:
                return False
    return True


# ---------------------------------------------------------------- synthetic frames
FX = W / (2.0 * math.tan(math.radians(FOV) / 2.0))
FACES = (((0, 1, 3, 2), (-1, 0, 0)), ((4, 6, 7, 5), (1, 0, 0)), ((0, 4, 5, 1), (0, -1, 0)),
         ((2, 3, 7, 6), (0, 1, 0)), ((1, 5, 7, 3), (0, 0, 1)))       # (corner indices, outward normal); no bottom


def _cuboid(cx, cy, l, w, h):
    return [np.array((cx + sx * l / 2, cy + sy * w / 2, z)) for sx in (-1, 1) for sy in (-1, 1) for z in (0, h)]


def _draw_cuboid(img, cam, inv, cam_pos, x, y, l, w, h, col):
    """Draw the camera-facing faces, clipped at the near plane (so close buildings don't vanish).
    Returns the bounding rect (x, y, w, h) of what was drawn, or None."""
    corners, drawn = _cuboid(x, y, l, w, h), []
    centre = np.array((x, y, h / 2))
    for idx, n in FACES:
        n = np.array(n, float)
        face_c = centre + n * np.array((l / 2, w / 2, h / 2))
        if np.dot(n, cam_pos - face_c) <= 0:                         # facing away from the camera
            continue
        pc = [(inv @ np.append(corners[i], 1.0))[:3] for i in idx]
        poly = []
        for a, b in zip(pc, pc[1:] + pc[:1]):                        # Sutherland-Hodgman, plane x_cam = 0.3
            if a[0] >= 0.3:
                poly.append(a)
            if (a[0] >= 0.3) != (b[0] >= 0.3):
                poly.append(a + (0.3 - a[0]) / (b[0] - a[0]) * (b - a))
        if len(poly) < 3:
            continue
        pix = np.array([(np.clip(W / 2 + FX * p[1] / p[0], -5000, 5000), np.clip(H / 2 - FX * p[2] / p[0], -5000, 5000))
                        for p in poly], np.int32)
        shade = 1.0 if n[2] > 0 else (0.82 if n[0] != 0 else 0.68)
        cv2.fillPoly(img, [pix], tuple(int(c * shade) for c in col))
        cv2.polylines(img, [pix], True, (20, 20, 20), 1)
        drawn.extend(pix.tolist())
    if not drawn:
        return None
    d = np.clip(np.array(drawn), (0, 0), (W - 1, H - 1))
    x0, y0 = d.min(axis=0)
    x1, y1 = d.max(axis=0)
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0)) if x1 > x0 and y1 > y0 else None


def _polyline(img, cam, pts):
    """Draw a world-space ground line, split wherever it leaves the camera's view."""
    seg = []
    for p in list(pts) + [None]:
        q = world_to_pixel(p, cam, W, H, FOV) if p is not None else None
        if q is not None:
            seg.append((int(q[0]), int(q[1])))
            continue
        if len(seg) > 1:
            cv2.polylines(img, [np.array(seg)], False, (110, 110, 110), 2)
        seg = []


def _road_lines(img, cam):
    j = SC["junction"]
    gap = (j["x"] - j["width"] / 2, j["x"] + j["width"] / 2) if j else (1e9, 1e9)
    for gy in (-ROAD_HALF, 0.0, ROAD_HALF):                            # ego road (gap at the junction)
        for x0, x1 in ((-15, gap[0]), (gap[1], 100)):
            _polyline(img, cam, [(x, gy, 0) for x in np.arange(x0, x1 + 0.01, 1.5)])
    if j:                                                              # cross road
        for gx in (gap[0], gap[1]):
            for y0, y1 in ((-60, -ROAD_HALF), (ROAD_HALF, 60)):
                _polyline(img, cam, [(gx, y, 0) for y in np.arange(y0, y1 + 0.01, 1.0)])
    c = SC.get("crossing")
    if c:                                                              # zebra stripes
        for y in np.arange(-ROAD_HALF + 0.5, ROAD_HALF, 1.0):
            _polyline(img, cam, [(x, y, 0) for x in np.linspace(c["x"] - c["width"] / 2, c["x"] + c["width"] / 2, 4)])


def render(cam, world, detections, show_ego):
    img = np.full((H, W, 3), (46, 38, 30), np.uint8)
    _road_lines(img, cam)
    colour = {"blocker": (95, 95, 95) if SC["blocker"]["cls"] else (88, 78, 70), "ped": (90, 130, 200)}
    items = [(o["x"], o["y"], o["l"], o["w"], o["h"], colour[o["id"]], o["id"]) for o in world.scene()]
    if show_ego:
        items.append((world.ex, 0.0, 4.5, 1.9, 1.5, (200, 140, 60), "ego"))
    cam_pos, inv = np.asarray(cam)[:3, 3], np.linalg.inv(np.asarray(cam))
    boxes = {}
    for x, y, l, w, h, col, oid in sorted(items, key=lambda i: -math.hypot(i[0] - cam_pos[0], i[1] - cam_pos[1])):
        rect = _draw_cuboid(img, cam, inv, cam_pos, x, y, l, w, h, col)
        if rect:
            boxes[oid] = rect
    dets = []                                                         # normalised boxes; the dashboard draws them
    for d in detections:
        if d["id"] in boxes:
            bx, by, bw, bh = boxes[d["id"]]
            box = [min(1.0, max(0.0, v)) for v in (bx / W, by / H, (bx + bw) / W, (by + bh) / H)]
            if box[2] - box[0] < 0.005 or box[3] - box[1] < 0.005:      # clipped off-screen
                continue
            dets.append({"cls": d["cls"], "conf": round(d["conf"], 2), "x": round(d["x"], 2), "y": round(d["y"], 2),
                         "box": [round(v, 4) for v in box]})
    return img, dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="intersection")
    ap.add_argument("--dash-port", type=int, default=8000)
    ap.add_argument("--transport", choices=["udp", "mqtt"], default="udp")
    ap.add_argument("--target-speed", type=float, default=35.0, help="km/h")
    ap.add_argument("--speed", type=float, default=1.0, help="wall-clock speed-up factor")
    ap.add_argument("--loop", action="store_true", help="restart the scenario automatically")
    a = ap.parse_args()
    global SC
    SC = SCENARIOS[a.scenario]

    world, pipe = World(a.target_speed), Pipeline(make_transport(a.transport))
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

    veh_cam = lambda: camera_matrix(world.ex + 1.5, 0.0, 1.6, 0.0, 0.0)
    rsu = SC["rsu"]
    rsu_cam = camera_matrix(rsu["x"], rsu["y"], rsu["z"], rsu["yaw"], rsu["pitch"])
    scenario = {k: SC[k] for k in ("name", "blocker", "junction", "crossing", "rsu")}
    state, last_push, last_res, tick, fps, t_prev = "CLEAR", 0.0, None, 0, 0.0, time.time()
    frames, boxes, ego_cam, rsu_bytes, last_onb = {}, {}, None, 0, []
    try:
        while True:
            if flags["reset"] or (a.loop and world.t > 22):
                world.reset(); pipe.reset(); flags["reset"] = False; state = "CLEAR"; last_onb = []
            if world.step(state):
                pipe.collision(world.t)
            if tick % 2 == 0:                                            # ~10 Hz detection
                rsu_d, onb_d = world.rsu_detect(), world.onboard_detect()
                cam_v = veh_cam()
                (img_r, box_r), (img_v, box_v) = render(rsu_cam, world, rsu_d, True), render(cam_v, world, onb_d, False)
                b64_r, rsu_bytes = to_b64_jpeg(img_r)
                b64_v, _ = to_b64_jpeg(img_v)
                frames, boxes = {"rsu": b64_r, "vehicle": b64_v}, {"rsu": box_r, "vehicle": box_v}
                ego_cam = {"matrix": cam_v, "width": W, "height": H, "fov": FOV, "ground_z": 0.0}
                pipe.rsu_publish(world.t, rsu_d, rsu_bytes)
                last_onb = onb_d
            time.sleep(0.002)                                            # let the UDP thread deliver
            last_res = pipe.vehicle_step(world.t, world.ego, last_onb)
            state = last_res["decision"]["state"]
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3); t_prev = now
            if now - last_push > 0.1:
                dash.push(pipe.make_state("mock", world.t, fps, world.ego, last_res, scenario, frames, boxes, ego_cam))
                last_push = now
            tick += 1
            time.sleep(max(0.0, DT / a.speed - 0.004))
    except KeyboardInterrupt:
        pass
    finally:
        dash.stop(); pipe.transport.close()


if __name__ == "__main__":
    main()
