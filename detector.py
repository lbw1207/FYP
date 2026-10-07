"""YOLO wrapper. Pass your CARLA-trained weights with --weights runs/detect/train/weights/best.pt"""
import cv2

DEFAULT_CLASSES = {"person", "bicycle", "motorcycle", "car", "truck", "bus"}   # COCO names


class Detector:
    def __init__(self, weights="yolov8n.pt", conf=0.35, imgsz=640, device=None, classes=DEFAULT_CLASSES):
        from ultralytics import YOLO                    # imported lazily: mock mode doesn't need it
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.allowed = set(classes) if classes else None
        self.names = self.model.names

    def detect(self, frame_bgr):
        res = self.model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
        out = []
        for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.cpu().numpy(),
                                          res.boxes.conf.cpu().numpy(),
                                          res.boxes.cls.cpu().numpy()):
            name = self.names[int(k)]
            if self.allowed is None or name in self.allowed:
                out.append({"cls": name, "conf": float(c), "box": (float(x1), float(y1), float(x2), float(y2))})
        return out


COLORS = {"vru": (0, 165, 255), "vehicle": (250, 165, 96), "other": (200, 200, 200)}   # BGR


def draw(frame, dets, extra_labels=None):
    """Draw boxes on `frame` (in place). dets need 'cls','conf','box'; extra_labels[i] optional text."""
    from fusion import group_of
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = map(int, d["box"])
        col = COLORS.get(group_of(d["cls"]), COLORS["other"])
        label = f'{d["cls"]} {d["conf"]:.2f}' + (f' | {extra_labels[i]}' if extra_labels and extra_labels[i] else "")
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), col, -1)
        cv2.putText(frame, label, (x1 + 3, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def to_b64_jpeg(frame, quality=70):
    import base64
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    raw = buf.tobytes()
    return base64.b64encode(raw).decode(), len(raw)
