# Perceptinet Dashboard

Live cooperative-perception dashboard for the Perceptinet FYP: shows the RSU camera feed, the
vehicle's onboard camera feed, a fused bird's-eye view, and the STOP / SLOW / CLEAR decision, all
updating in real time in a browser.

It runs in **two interchangeable modes** against the exact same dashboard and decision code:

- **`mock_sim.py`** — a small built-in kinematic scenario (no CARLA, no GPU, works on any laptop).
  Use this to develop and demo the dashboard UI itself.
- **`carla_sim.py`** — drives the same pipeline from a real CARLA simulation, using YOLO on the
  RSU and onboard camera feeds. This is the one that uses your CARLA-trained model.

Because both runners call the same `pipeline.py` / `fusion.py` / `dashboard_server.py`, anything
you validate in mock mode (the fusion logic, the decision thresholds, the dashboard layout) carries
over unchanged to the CARLA run — only the source of frames and detections differs.

## What it demonstrates

The blind-spot scenario from your slides: a truck parked on the roadside occludes a pedestrian from
the ego vehicle's onboard camera. An RSU camera (mounted overhead, on the opposite side of the
road) can see the pedestrian the whole time. The RSU's detector runs locally and sends **only the
detected object's class and world coordinates** over the network (UDP or MQTT) — never the image
itself — which is the "semantic communication, not raw video" point from your System Architecture
slide. The vehicle fuses that with its own onboard detections and reacts before the hazard is ever
visible on its own camera.

The dashboard's **Cooperative perception** toggle lets you switch RSU sharing on/off live, and the
**Injected V2I delay** slider lets you demonstrate what happens if the link is too slow — both are
the exact trade-offs your supervisor raised about latency.

## Quick start (no CARLA needed)

```bash
pip install fastapi "uvicorn[standard]" websockets opencv-python-headless numpy
python mock_sim.py
```

Then open **http://localhost:8000**. Toggle "Cooperative perception" off and restart the scenario —
watch the onboard-only run brake too late (or not at all) while the cooperative run slows down well
in advance.

Useful flags: `--speed 2` runs the scenario faster than real time; `--loop` restarts it
automatically; `--target-speed 40` changes the ego vehicle's cruising speed (km/h).

## Running against CARLA + your trained YOLO model

```bash
pip install carla==0.9.15 ultralytics fastapi "uvicorn[standard]" websockets opencv-python numpy
# match the carla version to your CARLA server's version

# 1) start the CARLA server first (CarlaUE4.sh / CarlaUE4.exe), then:
python carla_sim.py --weights runs/detect/train/weights/best.pt
```

Then open **http://localhost:8000**. Useful flags:

- `--weights yolov8n.pt` — use the stock pretrained model instead of your fine-tuned one
- `--town Town05` / `--spawn-index 12` — pick a different straight road segment if the default
  spawn doesn't suit your scene
- `--detect-every 2` — run YOLO every N ticks instead of every tick, if inference is the bottleneck
- `--device 0` — run YOLO on a specific GPU; `--device cpu` to force CPU
- `--transport mqtt --mqtt-host <broker>` — use MQTT instead of UDP for the V2I link (needs
  `pip install paho-mqtt` and a running broker, e.g. `mosquitto`)

`carla_sim.py` spawns the ego vehicle, a parked truck, a pedestrian hidden behind it, and the RSU
camera, all automatically, on a straight stretch of road it finds for you — you don't need to build
a custom CARLA map or place actors by hand.

## Files

| File | What it does |
|---|---|
| `fusion.py` | Camera geometry (pixel ↔ world ground-plane projection), late fusion of onboard + RSU detections, and the STOP/SLOW/CLEAR decision logic. Pure numpy, no CARLA/YOLO dependency — unit-testable on its own. |
| `transport.py` | The V2I link. `UdpTransport` (zero setup) or `MqttTransport` (needs a broker). Only a small JSON message crosses the wire. |
| `detector.py` | Thin `ultralytics.YOLO` wrapper (`Detector.detect`) plus drawing/encoding helpers used by both runners. |
| `pipeline.py` | Glues the above together: `rsu_publish()` (RSU side), `vehicle_step()` (vehicle side: receive → fuse → decide), `make_state()` (JSON snapshot sent to the dashboard). |
| `dashboard_server.py` | FastAPI + WebSocket server. Serves `static/index.html` and pushes/receives JSON. |
| `static/index.html` | The dashboard itself — dual camera feeds, bird's-eye canvas, decision banner, metrics, event log, and the cooperative/latency controls. Single self-contained file. |
| `mock_sim.py` | Runner #1: a small kinematic scenario standing in for CARLA, for UI development and demos without CARLA installed. |
| `carla_sim.py` | Runner #2: drives the real CARLA simulation + your YOLO model through the same pipeline. |

## Adapting it to your own YOLO classes / scenario

- If your CARLA-trained model uses class names other than COCO's (`person`, `car`, `truck`, ...),
  update the `GROUPS` dict at the top of `fusion.py` so the fusion/decision logic still knows which
  classes count as a "vulnerable road user" (`vru`) vs. a `vehicle`.
- Scenario geometry (truck distance, RSU height/angle, pedestrian trigger distance, etc.) is defined
  as named constants at the top of `carla_sim.py` / `mock_sim.py` — tune those rather than digging
  into the logic below them.
- The decision thresholds (braking distance, safety margins, corridor widths) are in
  `fusion.DecisionConfig` — pass a custom one into `Decider(cfg)` if you want to tune stopping
  behaviour without touching the decision algorithm itself.

## Known limitations to mention in your report

- `carla_sim.py` targets the CARLA 0.9.14–0.9.16 Python API; small API differences on other
  versions (e.g. `set_desired_speed` not existing pre-0.9.14) are already guarded with
  `hasattr` fallbacks, but double-check against whichever version your lab uses.
- The RSU→vehicle link is simulated over real UDP/MQTT sockets on localhost, not over CARLA's own
  network stack — this is intentional (it's what lets the same code run in mock mode with zero
  CARLA dependency), and matches how your architecture slide describes the V2I link as an external
  system, not part of the perception simulation itself.
- Detection runs on whichever machine runs the script; there's no on-device (Raspberry Pi / NCNN /
  Hailo) inference path in this dashboard, since this semester's deliverable is the simulation and
  dashboard, not the physical RSU (that's explicitly Semester 2 work per your System Architecture
  slide and the YOLO research report).
