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
    caution_half: float = 8.0   # VRUs this close to the path (e.g. waiting at a crossing) -> SLOW, unless walking away
    hold_s: float = 1.0         # keep STOP this long after the hazard clears (sim seconds)
    horizon: float = 4.0        # predict crossing VRUs at most this far ahead [s]
    min_cross_speed: float = 0.6  # lateral speed [m/s] above which a VRU counts as crossing


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
        yaw = math.radians(ego["yaw"])
        for o in objects:
            if group_of(o["cls"]) != "vru":
                continue
            f, l = to_ego_frame(ego["x"], ego["y"], ego["yaw"], o["x"], o["y"])
            if f <= 0 or f > slow_dist:
                continue
            # trajectory prediction: where will a crossing VRU be by the time the ego reaches it?
            lat, crossing = abs(l), False
            vl = -o.get("vx", 0.0) * math.sin(yaw) + o.get("vy", 0.0) * math.cos(yaw)
            if abs(vl) > c.min_cross_speed and l * vl < 0:
                l_pred = l + vl * min(f / max(v, 1.0), c.horizon)
                lat_pred = 0.0 if l * l_pred <= 0 else abs(l_pred)
                if lat_pred < lat:
                    lat, crossing = lat_pred, True
            walking_away = abs(vl) > c.min_cross_speed and l * vl > 0
            if lat > c.slow_half and (lat > c.caution_half or walking_away):
                continue
            in_stop = lat <= c.stop_half and (f <= stop_dist or (latched and f <= slow_dist))
            level = "STOP" if in_stop else "SLOW"
            better = hazard is None or (level == "STOP" and hazard["level"] != "STOP") or \
                     (level == hazard["level"] and f < hazard["f"])
            if better:
                hazard = {"level": level, "f": f, "l": l, "cls": o["cls"], "src": o["src"], "crossing": crossing}
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
            reason = f'{hazard["cls"].capitalize()} {"crossing " if hazard["crossing"] else ""}' \
                     f'{hazard["f"]:.0f} m ahead ({how})'
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
        while self.pending and sim_t >= self.pending[0]["timestamp"] + delay_s:
            self.latest = self.pending.popleft()
        return self.latest


# ----------------------------------------------------------------------------------
# Tracking: associates fused detections across steps, so the dashboard knows which
# objects are hidden from the onboard camera, since when, and where they are heading
# ----------------------------------------------------------------------------------
class Track:
    def __init__(self, tid, group):
        self.id, self.group, self.cls = tid, group, None
        self.x = self.y = None
        self.conf, self.last_t = 0.0, None
        self.first_onb = self.last_onb = self.first_rsu = self.last_rsu = None
        self.hist = deque(maxlen=15)            # (t, x, y) each time the measured position changes

    def observe(self, t, o):
        if self.x is None or (o["x"], o["y"]) != (self.x, self.y):
            self.hist.append((t, o["x"], o["y"]))
        self.cls, self.x, self.y, self.conf, self.last_t = o["cls"], o["x"], o["y"], o["conf"], t
        if o["src"] in ("onboard", "both"):
            self.last_onb = t
            self.first_onb = t if self.first_onb is None else self.first_onb
        if o["src"] in ("rsu", "both"):
            self.last_rsu = t
            self.first_rsu = t if self.first_rsu is None else self.first_rsu

    def hidden(self, t, grace=0.5):
        """Currently reported by the RSU, but not seen by the onboard camera for `grace` seconds."""
        return self.last_rsu is not None and t - self.last_rsu <= grace and \
            (self.last_onb is None or t - self.last_onb > grace)

    def velocity(self):
        """World-frame (vx, vy) from the oldest vs. newest few positions (averages out detector noise)."""
        if len(self.hist) < 10:
            return 0.0, 0.0
        k = len(self.hist) // 3
        a, b = list(self.hist)[:k], list(self.hist)[-k:]
        mean = lambda pts, i: sum(p[i] for p in pts) / len(pts)
        dt = mean(b, 0) - mean(a, 0)
        if dt < 0.6:
            return 0.0, 0.0
        return (mean(b, 1) - mean(a, 1)) / dt, (mean(b, 2) - mean(a, 2)) / dt


class Tracker:
    def __init__(self, gate=3.0, ttl=1.0):
        self.gate, self.ttl = gate, ttl
        self.reset()

    def reset(self):
        self.tracks, self._next = [], 1

    def update(self, t, fused):
        """Nearest-neighbour association (same class group). Tags each fused detection with "tid"."""
        free = list(self.tracks)
        for o in fused:
            g, best, best_d = group_of(o["cls"]), None, self.gate
            for tr in free:
                d = math.hypot(tr.x - o["x"], tr.y - o["y"])
                if tr.group == g and d < best_d:
                    best, best_d = tr, d
            if best is None:
                best = Track(self._next, g)
                self._next += 1
                self.tracks.append(best)
            else:
                free.remove(best)
            best.observe(t, o)
            o["tid"] = best.id
            o["vx"], o["vy"] = best.velocity()
        self.tracks = [tr for tr in self.tracks if t - tr.last_t <= self.ttl]
        return self.tracks
