"""
Shared vehicle/RSU pipeline used by BOTH carla_sim.py and mock_sim.py:
  RSU side    -> rsu_publish()   : detections (class + world x,y) -> V2I transport
  vehicle side-> vehicle_step()  : receive (+ optional injected delay) -> late fusion -> decision
  make_state()                   : JSON snapshot pushed to the dashboard
"""
import time
from collections import deque

from fusion import DelayBuffer, Decider, fuse, group_of, to_ego_frame


class Pipeline:
    def __init__(self, transport):
        self.transport = transport
        self.decider, self.delay_buf = Decider(), DelayBuffer()
        self.coop, self.latency_ms = True, 0.0
        self.msg_times, self.tx_ms = deque(maxlen=60), deque(maxlen=20)
        self.last_payload, self.last_frame_bytes, self.seq = 0, 0, 0
        self.reset()

    def reset(self):
        self.decider.reset()
        self.delay_buf.reset()
        self.t_rsu = self.t_onb = None            # first time a person was seen by RSU / onboard
        self.prev_state, self.collisions = "CLEAR", 0
        self.events = []

    # ---- RSU side ---------------------------------------------------------------
    def rsu_publish(self, t_sim, objects, frame_bytes=0):
        self.seq += 1
        self.last_frame_bytes = frame_bytes
        self.last_payload = self.transport.send({
            "src": "rsu", "seq": self.seq, "t_sim": t_sim,
            "objects": [{k: (round(o[k], 2) if isinstance(o[k], float) else o[k])
                         for k in ("cls", "conf", "x", "y")} for o in objects]})

    # ---- vehicle side -----------------------------------------------------------
    def vehicle_step(self, t_sim, ego, onboard):
        for m in self.transport.poll():
            self.delay_buf.push(m)
            self.tx_ms.append((m["recv_at"] - m["sent_at"]) * 1000.0)
            self.msg_times.append(m["recv_at"])
        delay_s = self.latency_ms / 1000.0
        latest = self.delay_buf.update(t_sim, delay_s)
        rsu = latest["objects"] if latest and t_sim - latest["t_sim"] <= delay_s + 0.8 else []

        self._track_first_seen(t_sim, ego, onboard, rsu)
        fused = fuse(onboard, rsu if self.coop else [])
        dec = self.decider.decide(t_sim, ego, fused)
        if dec["state"] != self.prev_state:
            lvl = {"STOP": "alert", "SLOW": "warn", "CLEAR": "ok"}[dec["state"]]
            self._event(t_sim, lvl, f'{dec["state"]}: {dec["reason"]}')
            self.prev_state = dec["state"]
        now = time.time()
        rate = sum(1 for x in self.msg_times if now - x <= 2.0) / 2.0
        tx = sum(self.tx_ms) / len(self.tx_ms) if self.tx_ms else 0.0
        lead = (self.t_onb - self.t_rsu) if (self.t_rsu is not None and self.t_onb is not None) else None
        return {"fused": fused, "decision": dec, "n_onboard": len(onboard), "n_rsu": len(rsu),
                "metrics": {"latency_ms": self.latency_ms + tx, "transport_ms": tx, "msgs_per_s": rate,
                            "payload_bytes": self.last_payload, "frame_bytes": self.last_frame_bytes,
                            "lead_time_s": lead}}

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

    def make_state(self, mode, t_sim, fps, ego, res, scenario, frames):
        """`scenario`: {"truck": {x,y,yaw,length,width}|None, "rsu": {x,y}}"""
        def rel(x, y):
            f, l = to_ego_frame(ego["x"], ego["y"], ego["yaw"], x, y)
            return round(f, 2), round(l, 2)
        objs = []
        for o in res["fused"]:
            f, l = rel(o["x"], o["y"])
            objs.append({"cls": o["cls"], "conf": round(o["conf"], 2), "src": o["src"], "f": f, "l": l})
        sc = {}
        if scenario.get("truck"):
            tr = scenario["truck"]
            f, l = rel(tr["x"], tr["y"])
            sc["truck"] = {"f": f, "l": l, "rel_yaw": tr["yaw"] - ego["yaw"],
                           "length": tr["length"], "width": tr["width"]}
        if scenario.get("rsu"):
            f, l = rel(scenario["rsu"]["x"], scenario["rsu"]["y"])
            sc["rsu"] = {"f": f, "l": l}
        events, self.events = self.events, []
        return {"mode": mode, "transport": self.transport.name, "t": round(t_sim, 2), "fps": round(fps, 1),
                "coop": self.coop, "latency_set_ms": self.latency_ms,
                "ego": {"speed_kmh": round(ego["speed"] * 3.6, 1)},
                "objects": objs, "scenario": sc, "decision": res["decision"], "collisions": self.collisions,
                "metrics": dict(res["metrics"], n_onboard=res["n_onboard"], n_rsu=res["n_rsu"],
                                n_fused=len(objs)),
                "events": events, "frames": frames}
