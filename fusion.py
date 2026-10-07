"""
Perceptinet - geometry, late fusion and the vehicle-side decision logic.

Pure numpy / stdlib: no CARLA or YOLO imports, so it can be unit-tested and reused
by both the CARLA runner and the mock runner.

Coordinate convention = CARLA / Unreal:  x forward, y RIGHT, z up, yaw in degrees
(positive yaw turns clockwise when seen from above).
"""
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

# ----------------------------------------------------------------------------------
# Class groups (add your own class names here if your CARLA-trained model uses them)
# ----------------------------------------------------------------------------------
GROUPS = {
    "person": "vru", "pedestrian": "vru", "walker": "vru",
    "bicycle": "vru", "cyclist": "vru", "motorcycle": "vru",
    "car": "vehicle", "van": "vehicle", "truck": "vehicle", "bus": "vehicle", "vehicle": "vehicle",
}


def group_of(cls: str) -> str:
    return GROUPS.get(cls, cls)


# ----------------------------------------------------------------------------------
# Camera geometry  (ground-plane homography: pixel <-> shared world (x, y) grid)
# ----------------------------------------------------------------------------------
def camera_matrix(x, y, z, yaw_deg, pitch_deg):
    """4x4 camera->world matrix. Camera axes: x forward, y right, z up (CARLA sensor frame)."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    f = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    r = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    u = np.array([-math.sin(pitch) * math.cos(yaw), -math.sin(pitch) * math.sin(yaw), math.cos(pitch)])
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = f, r, u, (x, y, z)
    return m


def _intrinsics(width, height, fov_deg):
    fx = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return fx, fx, width / 2.0, height / 2.0


def pixel_to_ground(u, v, cam_matrix, width, height, fov_deg, ground_z=0.0):
    """Project pixel (u, v) onto the plane z = ground_z. Returns world (x, y) or None."""
    fx, fy, cx, cy = _intrinsics(width, height, fov_deg)
    d_cam = np.array([1.0, (u - cx) / fx, -(v - cy) / fy, 0.0])
    m = np.asarray(cam_matrix, dtype=float)
    d = (m @ d_cam)[:3]
    o = m[:3, 3]
    if d[2] >= -1e-6:                       # ray never reaches the ground (above horizon)
        return None
    t = (ground_z - o[2]) / d[2]
    if t <= 0:
        return None
    p = o + t * d
    return float(p[0]), float(p[1])


def world_to_pixel(p_world, cam_matrix, width, height, fov_deg):
    """Inverse projection (used by the mock renderer). Returns (u, v, depth) or None."""
    fx, fy, cx, cy = _intrinsics(width, height, fov_deg)
    pc = np.linalg.inv(np.asarray(cam_matrix, dtype=float)) @ np.array([*p_world, 1.0])
    if pc[0] < 0.3:
        return None
    return float(cx + fx * pc[1] / pc[0]), float(cy - fy * pc[2] / pc[0]), float(pc[0])


def to_ego_frame(ex, ey, eyaw_deg, x, y):
    """World (x, y) -> (forward, right) distances relative to the ego vehicle."""
    yaw = math.radians(eyaw_deg)
    dx, dy = x - ex, y - ey
    return (dx * math.cos(yaw) + dy * math.sin(yaw),
            -dx * math.sin(yaw) + dy * math.cos(yaw))


# ----------------------------------------------------------------------------------
# Late fusion: merge onboard + RSU detections (both already in world coordinates)
# ----------------------------------------------------------------------------------
def fuse(onboard, rsu, merge_dist=2.0):
    """Each detection: {"cls","conf","x","y"}. Returns fused list with a "src" tag:
    'onboard' (only the vehicle sees it), 'rsu' (only the RSU sees it -> occluded
    for the vehicle) or 'both'."""
    fused = [dict(o, src="onboard") for o in onboard]
    for r in rsu:
        best, best_d = None, merge_dist
        for f in fused:
            if f["src"] == "both" or group_of(f["cls"]) != group_of(r["cls"]):
                continue
            d = math.hypot(f["x"] - r["x"], f["y"] - r["y"])
            if d < best_d:
                best, best_d = f, d
        if best is not None:
            w1, w2 = best["conf"], r["conf"]
            best["x"] = (best["x"] * w1 + r["x"] * w2) / (w1 + w2)
            best["y"] = (best["y"] * w1 + r["y"] * w2) / (w1 + w2)
            best["conf"] = max(w1, w2)
            best["src"] = "both"
        else:
            fused.append(dict(r, src="rsu"))
    return fused


# ----------------------------------------------------------------------------------
# Decision logic (what the dashboard shows and what CARLA executes)
# ----------------------------------------------------------------------------------
@dataclass
class DecisionConfig:
    decel: float = 5.0          # assumed braking deceleration [m/s^2]
    margin: float = 6.0         # safety margin [m]
    slow_factor: float = 2.0    # slow zone = stop_dist * factor + slow_extra
    slow_extra: float = 8.0
    stop_half: float = 2.0      # half-width of the STOP corridor [m]
    slow_half: float = 4.0      # half-width of the SLOW corridor (kerb-side pedestrians) [m]
    hold_s: float = 1.0         # keep STOP this long after the hazard clears (sim seconds)


class Decider:
    def __init__(self, cfg: DecisionConfig = None):
        self.cfg = cfg or DecisionConfig()
        self.reset()

    def reset(self):
        self.state = "CLEAR"
        self._stop_until = -1.0

    def decide(self, t, ego, objects):
        c, v = self.cfg, max(ego["speed"], 0.0)
        stop_dist = v * v / (2 * c.decel) + c.margin
        slow_dist = stop_dist * c.slow_factor + c.slow_extra
        latched = self.state == "STOP"
        raw, hazard = "CLEAR", None
        for o in objects:
            if group_of(o["cls"]) != "vru":
                continue
            f, l = to_ego_frame(ego["x"], ego["y"], ego["yaw"], o["x"], o["y"])
            if f <= 0 or abs(l) > c.slow_half or f > slow_dist:
                continue
            in_stop = abs(l) <= c.stop_half and (f <= stop_dist or (latched and f <= slow_dist))
            level = "STOP" if in_stop else "SLOW"
            better = hazard is None or (level == "STOP" and hazard["level"] != "STOP") or \
                     (level == hazard["level"] and f < hazard["f"])
            if better:
                hazard = {"level": level, "f": f, "l": l, "cls": o["cls"], "src": o["src"]}
        if hazard:
            raw = hazard["level"]
        if raw == "STOP":
            self._stop_until = t + c.hold_s
        elif t < self._stop_until:
            raw = "STOP"
        self.state = raw
        reason = "No vulnerable road user in the path"
        if hazard:
            how = {"rsu": "seen by RSU only", "onboard": "seen onboard",
                   "both": "seen by RSU + onboard"}[hazard["src"]]
            reason = f'{hazard["cls"].capitalize()} {hazard["f"]:.0f} m ahead ({how})'
        elif raw == "STOP":
            reason = "Holding stop while the path clears"
        return {"state": raw, "reason": reason, "hazard": hazard,
                "stop_dist": stop_dist, "slow_dist": slow_dist,
                "stop_half": c.stop_half, "slow_half": c.slow_half}


# ----------------------------------------------------------------------------------
# Simulation-time V2I delay (lets you demo the latency problem your supervisor raised)
# ----------------------------------------------------------------------------------
class DelayBuffer:
    def __init__(self):
        self.pending, self.latest = deque(), None

    def reset(self):
        self.pending.clear()
        self.latest = None

    def push(self, msg):
        self.pending.append(msg)

    def update(self, sim_t, delay_s):
        while self.pending and sim_t >= self.pending[0]["t_sim"] + delay_s:
            self.latest = self.pending.popleft()
        return self.latest
