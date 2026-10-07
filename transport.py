"""
V2I link between the RSU side and the vehicle side.
Only a small JSON message (object class + world coordinates) travels - never the image.
UDP (default, zero setup) or MQTT (needs a broker, e.g. `mosquitto`).
"""
import json
import queue
import socket
import threading
import time


class UdpTransport:
    name = "UDP"

    def __init__(self, host="127.0.0.1", port=5005):
        self.addr = (host, port)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx.bind(self.addr)
        self.rx.settimeout(0.2)
        self.q = queue.Queue()
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def send(self, msg: dict) -> int:
        data = json.dumps(dict(msg, sent_at=time.time()), separators=(",", ":")).encode()
        self.tx.sendto(data, self.addr)
        return len(data)

    def _loop(self):
        while self._run:
            try:
                data, _ = self.rx.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            msg = json.loads(data)
            msg["recv_at"], msg["bytes"] = time.time(), len(data)
            self.q.put(msg)

    def poll(self):
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        self._run = False
        self.rx.close()
        self.tx.close()


class MqttTransport:
    name = "MQTT"

    def __init__(self, host="127.0.0.1", port=1883, topic="perceptinet/rsu/detections"):
        import paho.mqtt.client as mqtt
        self.topic, self.q = topic, queue.Queue()
        try:                                              # paho-mqtt >= 2.0
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:                            # paho-mqtt 1.x
            self.client = mqtt.Client()
        self.client.on_connect = lambda c, *a: c.subscribe(self.topic)
        self.client.on_message = self._on_message
        self.client.connect(host, port, 60)
        self.client.loop_start()

    def _on_message(self, client, userdata, m):
        msg = json.loads(m.payload)
        msg["recv_at"], msg["bytes"] = time.time(), len(m.payload)
        self.q.put(msg)

    def send(self, msg: dict) -> int:
        data = json.dumps(dict(msg, sent_at=time.time()), separators=(",", ":"))
        self.client.publish(self.topic, data, qos=0)
        return len(data)

    def poll(self):
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()


def make_transport(kind="udp", host="127.0.0.1", port=None):
    if kind == "mqtt":
        return MqttTransport(host=host, port=port or 1883)
    return UdpTransport(port=port or 5005)
