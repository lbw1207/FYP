#!/usr/bin/env python
"""
Perceptinet dashboard WITHOUT CARLA: a tiny kinematic replica of the blind-spot scenario.
Uses exactly the same pipeline (transport -> late fusion -> decision) and the same dashboard,
so you can develop/demo the UI on any laptop and compare it with the CARLA run.

    python mock_sim.py            ->  http://localhost:8000
"""
import argparse
import math
import random
import time
from collections import deque

import cv2
import numpy as np

from dashboard_server import DashboardServer
from detector import draw, to_b64_jpeg
from fusion import camera_matrix, world_to_pixel, group_of
from pipeline import Pipeline
from transport import make_transport

W, H, FOV, DT = 640, 360, 90, 0.05
TRUCK = {"x": 40.0, "y": 2.1, "yaw": 0.0, "length": 9.0, "width": 2.5, "height": 3.6}
RSU = {"x": 44.0, "y": -6.0, "z": 8.0, "yaw": 90.0, "pitch": -35.0}
PED_START, TRIGGER_DIST, CROSS_TIME, PED_SPEED = (48.5, 3.0), 22.0, 6.0, 1.5


class World:
    def __init__(self, target_kmh):
        self.target = target_kmh / 3.6
        self.reset()

    def reset(self):
        self.t, self.ex, self.ev = 0.0, 0.0, 0.0
        self.px, self.py, self.walk_t = *PED_START, None
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
        if self.walk_t is None and math.hypot(self.px - self.ex, self.py) < TRIGGER_DIST:
            self.walk_t = 0.0
        if self.walk_t is not None and self.walk_t < CROSS_TIME:
            self.py -= PED_SPEED * DT
            self.walk_t += DT
        self.t += DT
        newly_hit = (not self.hit) and abs(self.px - self.ex) < 2.5 and abs(self.py) < 1.2
        self.hit = self.hit or newly_hit
        return newly_hit

    # ---- what each camera can "detect" (stand-in for YOLO) -------------------
    def objects(self):
        return [{"cls": "truck", "x": TRUCK["x"], "y": TRUCK["y"], "h": TRUCK["height"], "l": TRUCK["length"], "w": TRUCK["width"], "id": "truck"},
                {"cls": "person", "x": self.px, "y": self.py, "h": 1.75, "l": 0.5, "w": 0.5, "id": "ped"}]

    def onboard_detect(self):
        out = []
        for o in self.objects():
            f = o["x"] - self.ex
            rng = 38.0 if o["cls"] == "person" else 70.0            # small objects are detected later
            if not (1 < f < rng) or abs(o["y"]) > f:                 # outside the 90-degree FOV
                continue
            if o["id"] != "truck" and _occluded((self.ex + 1.5, 0.0), (o["x"], o["y"])):
                continue
            if random.random() < 0.05:
                continue
            out.append({"cls": o["cls"], "conf": random.uniform(.6, .9), "id": o["id"],
                        "x": o["x"] + random.gauss(0, .25), "y": o["y"] + random.gauss(0, .25)})
        return out

    def rsu_detect(self):
        out = []
        for o in self.objects():
            if math.hypot(o["x"] - RSU["x"], o["y"] - RSU["y"]) < 45 and random.random() > 0.03:
                out.append({"cls": o["cls"], "conf": random.uniform(.65, .92), "id": o["id"],
                            "x": o["x"] + random.gauss(0, .25), "y": o["y"] + random.gauss(0, .25)})
        return out


def _occluded(p0, p1):
    """2-D segment vs. the truck's footprint rectangle (slab test)."""
    xmin, xmax = TRUCK["x"] - TRUCK["length"] / 2, TRUCK["x"] + TRUCK["length"] / 2
    ymin, ymax = TRUCK["y"] - TRUCK["width"] / 2, TRUCK["y"] + TRUCK["width"] / 2
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
def _cuboid(cx, cy, l, w, h):
    return [(cx + sx * l / 2, cy + sy * w / 2, z) for sx in (-1, 1) for sy in (-1, 1) for z in (0, h)]


def render(cam, world, detections, show_ego):
    img = np.full((H, W, 3), (46, 38, 30), np.uint8)
    for gy in (-5.25, 0.0, 5.25):                                     # lane lines
        pts = [world_to_pixel((x, gy, 0), cam, W, H, FOV) for x in np.arange(-15, 100, 1.5)]
        pts = [(int(p[0]), int(p[1])) for p in pts if p]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts)], False, (110, 110, 110), 2)
    items = [(o["x"], o["y"], o["l"], o["w"], o["h"], (95, 95, 95) if o["id"] == "truck" else (90, 130, 200), o["id"]) for o in world.objects()]
    if show_ego:
        items.append((world.ex, 0.0, 4.5, 1.9, 1.5, (200, 140, 60), "ego"))
    cam_pos = np.asarray(cam)[:3, 3]
    boxes = {}
    for x, y, l, w, h, col, oid in sorted(items, key=lambda i: -math.hypot(i[0] - cam_pos[0], i[1] - cam_pos[1])):
        pts = [world_to_pixel(p, cam, W, H, FOV) for p in _cuboid(x, y, l, w, h)]
        if any(p is None for p in pts):
            continue
        poly = np.array([(int(p[0]), int(p[1])) for p in pts])
        hull = cv2.convexHull(poly)
        cv2.fillConvexPoly(img, hull, col)
        cv2.polylines(img, [hull], True, (20, 20, 20), 1)
        boxes[oid] = cv2.boundingRect(hull)
    dets = []
    for d in detections:
        if d["id"] in boxes:
            bx, by, bw, bh = boxes[d["id"]]
            dets.append({"cls": d["cls"], "conf": d["conf"], "box": (bx, by, bx + bw, by + bh)})
    return draw(img, dets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dash-port", type=int, default=8000)
    ap.add_argument("--transport", choices=["udp", "mqtt"], default="udp")
    ap.add_argument("--target-speed", type=float, default=35.0, help="km/h")
    ap.add_argument("--speed", type=float, default=1.0, help="wall-clock speed-up factor")
    ap.add_argument("--loop", action="store_true", help="restart the scenario automatically")
    a = ap.parse_args()

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
    rsu_cam = camera_matrix(RSU["x"], RSU["y"], RSU["z"], RSU["yaw"], RSU["pitch"])
    scenario = {"truck": TRUCK, "rsu": RSU}
    state, last_push, last_res, tick, fps, t_prev = "CLEAR", 0.0, None, 0, 0.0, time.time()
    frames, rsu_bytes, last_onb = {}, 0, []
    try:
        while True:
            if flags["reset"] or (a.loop and world.t > 22):
                world.reset(); pipe.reset(); flags["reset"] = False; state = "CLEAR"; last_onb = []
            if world.step(state):
                pipe.collision(world.t)
            if tick % 2 == 0:                                            # ~10 Hz detection
                rsu_d, onb_d = world.rsu_detect(), world.onboard_detect()
                img_r, img_v = render(rsu_cam, world, rsu_d, True), render(veh_cam(), world, onb_d, False)
                b64_r, rsu_bytes = to_b64_jpeg(img_r)
                b64_v, _ = to_b64_jpeg(img_v)
                frames = {"rsu": b64_r, "vehicle": b64_v}
                pipe.rsu_publish(world.t, rsu_d, rsu_bytes)
                last_onb = onb_d
            time.sleep(0.002)                                            # let the UDP thread deliver
            last_res = pipe.vehicle_step(world.t, world.ego, last_onb)
            state = last_res["decision"]["state"]
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3); t_prev = now
            if now - last_push > 0.1:
                dash.push(pipe.make_state("mock", world.t, fps, world.ego, last_res, scenario, frames))
                last_push = now
            tick += 1
            time.sleep(max(0.0, DT / a.speed - 0.004))
    except KeyboardInterrupt:
        pass
    finally:
        dash.stop(); pipe.transport.close()


if __name__ == "__main__":
    main()
