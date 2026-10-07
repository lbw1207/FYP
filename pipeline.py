"""
Shared vehicle/RSU pipeline used by BOTH carla_sim.py and mock_sim.py:
  RSU side    -> rsu_publish()   : detections -> object-level V2I message (never the image)
  vehicle side-> vehicle_step()  : receive (+ optional injected delay) -> late fusion -> tracking -> decision
  make_state()                   : JSON snapshot pushed to the dashboard (Driver mode + System mode)
"""
import math
import time
from collections import deque

from fusion import DelayBuffer, Decider, Tracker, fuse, group_of, to_ego_frame, world_to_pixel

RSU_ID = "RSU-01"
GHOST_SIZE = {"vru": (0.7, 1.8), "vehicle": (2.0, 1.6)}      # rough (width, height) [m] of a hidden object
NAMES = {"person": "Pedestrian", "pedestrian": "Pedestrian", "walker": "Pedestrian", "bicycle": "Cyclist",
         "cyclist": "Cyclist", "motorcycle": "Motorcyclist", "car": "Vehicle", "van": "Vehicle",
         "vehicle": "Vehicle", "truck": "Truck", "bus": "Bus"}


def _decode(msg):
    """V2I message objects -> internal detections."""
    return [{"cls": o["type"], "conf": o["confidence"], "x": o["position"][0], "y": o["position"][1]}
            for o in msg["objects"]]


def _mean_conf(objs):
    return sum(o["conf"] for o in objs) / len(objs) if objs else None


class Pipeline:
    def __init__(self, transport, rsu_id=RSU_ID):
        self.transport, self.rsu_id = transport, rsu_id
        self.decider, self.delay_buf, self.tracker = Decider(), DelayBuffer(), Tracker()
        self.coop, self.latency_ms = True, 0.0
        self.msg_times, self.tx_ms = deque(maxlen=60), deque(maxlen=20)
        self.last_payload, self.last_frame_bytes, self.seq = 0, 0, 0
        self.rx_seen, self.last_msg, self.fusion_ms = {}, None, None
        self.reset()

    def reset(self):
        self.decider.reset()
        self.delay_buf.reset()
        self.tracker.reset()
        self.t_rsu = self.t_onb = None            # first time a person was seen by RSU / onboard
        self.prev_state, self.collisions = "CLEAR", 0
        self.events = []

    # ---- RSU side ---------------------------------------------------------------
    def rsu_publish(self, t_sim, objects, frame_bytes=0):
        self.seq += 1
        self.last_frame_bytes = frame_bytes
        self.last_payload = self.transport.send({
            "rsu_id": self.rsu_id, "seq": self.seq, "timestamp": round(t_sim, 3),
            "objects": [{"type": o["cls"], "position": [round(o["x"], 2), round(o["y"], 2)],
                         "confidence": round(o["conf"], 2)} for o in objects]})

    # ---- vehicle side -----------------------------------------------------------
    def vehicle_step(self, t_sim, ego, onboard):
        for m in self.transport.poll():
            self.delay_buf.push(m)
            self.tx_ms.append((m["recv_at"] - m["sent_at"]) * 1000.0)
            self.msg_times.append(m["recv_at"])
            self.rx_seen[m.get("rsu_id", "rsu")] = m["recv_at"]
            self.last_msg = {k: v for k, v in m.items() if k not in ("sent_at", "recv_at", "bytes")}
        delay_s = self.latency_ms / 1000.0
        latest = self.delay_buf.update(t_sim, delay_s)
        rsu = _decode(latest) if latest and t_sim - latest["timestamp"] <= delay_s + 0.8 else []

        self._track_first_seen(t_sim, ego, onboard, rsu)
        t0 = time.perf_counter()
        fused = fuse(onboard, rsu if self.coop else [])
        tracks = self.tracker.update(t_sim, fused)
        dec = self.decider.decide(t_sim, ego, fused)
        ms = (time.perf_counter() - t0) * 1000.0
        self.fusion_ms = ms if self.fusion_ms is None else 0.9 * self.fusion_ms + 0.1 * ms
        if dec["state"] != self.prev_state:
            lvl = {"STOP": "alert", "SLOW": "warn", "CLEAR": "ok"}[dec["state"]]
            self._event(t_sim, lvl, f'{dec["state"]}: {dec["reason"]}')
            self.prev_state = dec["state"]
        now = time.time()
        rate = sum(1 for x in self.msg_times if now - x <= 2.0) / 2.0
        tx = sum(self.tx_ms) / len(self.tx_ms) if self.tx_ms else 0.0
        lead = (self.t_onb - self.t_rsu) if (self.t_rsu is not None and self.t_onb is not None) else None
        return {"fused": fused, "tracks": tracks, "decision": dec, "n_onboard": len(onboard), "n_rsu": len(rsu),
                "metrics": {"latency_ms": self.latency_ms + tx, "transport_ms": tx, "msgs_per_s": rate,
                            "payload_bytes": self.last_payload, "frame_bytes": self.last_frame_bytes,
                            "lead_time_s": lead, "t_rsu_detect": self.t_rsu, "t_onb_detect": self.t_onb,
                            "fusion_ms": self.fusion_ms, "conf_onboard": _mean_conf(onboard),
                            "conf_rsu": _mean_conf(rsu)}}

    def _track_first_seen(self, t, ego, onboard, rsu):
        def nearest_person(objs):
            best = None
            for o in objs:
                if group_of(o["cls"]) == "vru":
                    f, _ = to_ego_frame(ego["x"], ego["y"], ego["yaw"], o["x"], o["y"])
                    if f > 0 and (best is None or f < best):
                        best = f
            return best
        d_rsu, d_onb = nearest_person(rsu), nearest_person(onboard)
        if d_rsu is not None and self.t_rsu is None:
            self.t_rsu = t
            self._event(t, "info", f"RSU detected a pedestrian {d_rsu:.0f} m ahead"
                                   + (" - not yet visible onboard" if d_onb is None else ""))
        if d_onb is not None and self.t_onb is None:
            self.t_onb = t
            lead = f" (RSU warned {t - self.t_rsu:.1f} s earlier)" if self.t_rsu is not None else ""
            self._event(t, "info", f"Onboard camera now sees the pedestrian at {d_onb:.0f} m{lead}")

    def _event(self, t, level, msg):
        self.events.append({"t": round(t, 1), "level": level, "msg": msg})

    def collision(self, t):
        self.collisions += 1
        self._event(t, "alert", "COLLISION with pedestrian")

    # ---- dashboard snapshot -----------------------------------------------------
    @staticmethod
    def _hidden(t, tracks, grace=0.5):
        """Tracks the RSU reports but the onboard camera cannot see (occluded / out of view)."""
        seen = [tr for tr in tracks if tr.last_onb is not None and t - tr.last_onb <= grace]
        return [tr for tr in tracks if tr.hidden(t, grace) and t - tr.first_rsu >= 0.2 and not any(
            s.group == tr.group and math.hypot(s.x - tr.x, s.y - tr.y) < 4.0 for s in seen)]

    @staticmethod
    def _ghost(tr, cam):
        """Project a hidden object into the ego camera image ("virtual perception"). Normalised box or None."""
        w, h = GHOST_SIZE.get(tr.group, (1.0, 1.5))
        gz, W, H = cam.get("ground_z", 0.0), cam["width"], cam["height"]
        args = (cam["matrix"], W, H, cam["fov"])
        bot, top = world_to_pixel((tr.x, tr.y, gz), *args), world_to_pixel((tr.x, tr.y, gz + h), *args)
        if bot is None or top is None:
            return None
        ph = bot[1] - top[1]
        u1, u2, v1, v2 = (bot[0] - ph * w / h / 2) / W, (bot[0] + ph * w / h / 2) / W, top[1] / H, bot[1] / H
        if u2 < 0 or u1 > 1 or v2 < 0 or v1 > 1:
            return None
        return [round(min(1.0, max(0.0, v)), 4) for v in (u1, v1, u2, v2)]

    @staticmethod
    def _alert(t, tr, f, l, hidden, approaching, level):
        name, side = NAMES.get(tr.cls, tr.cls.capitalize()), ("right" if l > 0 else "left")
        vru = tr.group == "vru"
        if hidden and approaching:
            title = f"{name} approaching" if vru else f"{name} entering your path"
        elif hidden:
            title = f"{name} behind obstruction" if vru else f"{name} detected behind obstruction"
        else:
            title = f"{name} ahead"
        detail = (f"Hidden from your camera on the {side} - reported by the roadside unit" if hidden
                  else "In your path - now visible to your camera")
        lead = None
        if tr.first_rsu is not None and tr.first_onb is not None and tr.first_onb > tr.first_rsu:
            lead = round(tr.first_onb - tr.first_rsu, 1)
        return {"tid": tr.id, "cls": tr.cls, "kind": tr.group, "title": title, "detail": detail,
                "dist": round(f), "side": side, "hidden": hidden, "approaching": approaching, "level": level,
                "lead_s": lead, "since_s": round(t - tr.first_rsu, 1) if hidden and tr.first_rsu is not None else None}

    def _driver_view(self, t, ego, res, hidden, cam, linked):
        """Driver mode: only actionable info - what is hidden, how far, and what to do."""
        dec = res["decision"]
        hz, yaw = dec["hazard"], math.radians(ego["yaw"])
        alerts, ghosts = [], []
        for tr in res["tracks"]:
            f, l = to_ego_frame(ego["x"], ego["y"], ego["yaw"], tr.x, tr.y)
            is_hidden = tr in hidden
            is_hazard = hz is not None and tr.group == group_of(hz["cls"]) and \
                abs(f - hz["f"]) < 2.5 and abs(l - hz["l"]) < 2.5
            if not (is_hazard or (is_hidden and 0 < f <= 80 and abs(l) <= 15)):
                continue
            vx, vy = tr.velocity()
            vl = -vx * math.sin(yaw) + vy * math.cos(yaw)              # lateral speed in the ego frame
            approaching = l * vl < 0 and abs(vl) > 0.6 and abs(l) > 1.0
            alerts.append(self._alert(t, tr, f, l, is_hidden, approaching, hz["level"] if is_hazard else "INFO"))
            if is_hidden and cam:
                box = self._ghost(tr, cam)
                if box:
                    ghosts.append({"tid": tr.id, "cls": tr.cls, "conf": round(tr.conf, 2), "dist": round(f), "box": box})
        rank = {"STOP": 0, "SLOW": 1, "INFO": 2}
        alerts.sort(key=lambda a: (rank[a["level"]], not (a["hidden"] and a["approaching"]), a["dist"]))
        return {"action": dec["state"], "reason": dec["reason"],
                "assist": "off" if not self.coop else ("on" if linked else "lost"),
                "alerts": alerts[:3], "ghosts": ghosts}

    def make_state(self, mode, t_sim, fps, ego, res, scenario, frames, boxes=None, ego_cam=None):
        """`scenario`: {"name", "blocker": {x,y,yaw,length,width,label}|None, "rsu": {x,y},
                     "junction": {x,y,yaw,width}|None  (cross road of a blind intersection),
                     "crossing": {x,y,width}|None      (zebra crossing over the ego road)}
        `boxes`:    {"vehicle"|"rsu": [{"cls","conf","x","y","box":[u1,v1,u2,v2] normalised 0-1}]}
        `ego_cam`:  {"matrix","width","height","fov","ground_z"} of the frame in `frames["vehicle"]`,
                    used to project RSU-only objects into the driver's view."""
        def rel(x, y):
            f, l = to_ego_frame(ego["x"], ego["y"], ego["yaw"], x, y)
            return round(f, 2), round(l, 2)
        hidden = [tr for tr in self._hidden(t_sim, res["tracks"]) if rel(tr.x, tr.y)[0] > 0]   # ahead only
        hidden_ids = {tr.id for tr in hidden}
        objs = []
        for o in res["fused"]:
            f, l = rel(o["x"], o["y"])
            objs.append({"cls": o["cls"], "conf": round(o["conf"], 2), "src": o["src"], "f": f, "l": l,
                         "hidden": o.get("tid") in hidden_ids})
        sc = {}
        if scenario.get("blocker"):
            b = scenario["blocker"]
            f, l = rel(b["x"], b["y"])
            sc["blocker"] = {"f": f, "l": l, "rel_yaw": b["yaw"] - ego["yaw"], "length": b["length"],
                             "width": b["width"], "label": b.get("label", "obstruction")}
        if scenario.get("junction"):
            j = scenario["junction"]
            f, l = rel(j["x"], j["y"])
            sc["junction"] = {"f": f, "l": l, "rel_yaw": j.get("yaw", 0.0) - ego["yaw"], "width": j["width"]}
        if scenario.get("crossing"):
            c = scenario["crossing"]
            f, l = rel(c["x"], c["y"])
            sc["crossing"] = {"f": f, "l": l, "width": c["width"]}
        if scenario.get("rsu"):
            f, l = rel(scenario["rsu"]["x"], scenario["rsu"]["y"])
            sc["rsu"] = {"f": f, "l": l}
        boxes = {k: [dict(b, f=rel(b["x"], b["y"])[0]) for b in v] for k, v in (boxes or {}).items()}
        for b in boxes.get("rsu", []):
            b["hidden"] = any(math.hypot(tr.x - b["x"], tr.y - b["y"]) < 3.0 for tr in hidden)
        now = time.time()
        active = sorted(k for k, v in self.rx_seen.items() if now - v <= 1.0)
        events, self.events = self.events, []
        return {"mode": mode, "transport": self.transport.name, "t": round(t_sim, 2), "fps": round(fps, 1),
                "coop": self.coop, "latency_set_ms": self.latency_ms,
                "status": {"coop": self.coop, "active_rsus": len(active), "rsu_ids": active,
                           "v2i": "Connected" if active else "No signal",
                           "scenario": scenario.get("name", "-"), "net_delay_ms": self.latency_ms},
                "ego": {"speed_kmh": round(ego["speed"] * 3.6, 1)},
                "objects": objs, "scenario": sc, "decision": res["decision"], "collisions": self.collisions,
                "hidden": [dict(zip(("f", "l"), rel(tr.x, tr.y)), cls=tr.cls, conf=round(tr.conf, 2)) for tr in hidden],
                "driver": self._driver_view(t_sim, ego, res, hidden, ego_cam, bool(active)),
                "metrics": dict(res["metrics"], n_onboard=res["n_onboard"], n_rsu=res["n_rsu"],
                                n_fused=len(objs)),
                "v2i_msg": self.last_msg, "boxes": boxes, "events": events, "frames": frames}
