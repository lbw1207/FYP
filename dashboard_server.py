"""FastAPI + WebSocket server. The simulator calls push(state); the browser can send commands back."""
import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class DashboardServer:
    def __init__(self, port=8000, on_command=None):
        self.port, self.on_command = port, on_command
        self.clients, self.loop, self.pending = set(), None, 0

        @asynccontextmanager
        async def lifespan(app):
            self.loop = asyncio.get_running_loop()
            yield

        self.app = FastAPI(lifespan=lifespan)

        @self.app.get("/")
        async def index():
            return FileResponse(os.path.join(STATIC, "index.html"), headers={"Cache-Control": "no-store"})

        @self.app.websocket("/ws")
        async def ws_endpoint(ws: WebSocket):
            await ws.accept()
            self.clients.add(ws)
            try:
                while True:
                    cmd = json.loads(await ws.receive_text())
                    if self.on_command:
                        self.on_command(cmd)
            except (WebSocketDisconnect, RuntimeError):
                pass
            finally:
                self.clients.discard(ws)

    def start(self):
        cfg = uvicorn.Config(self.app, host="0.0.0.0", port=self.port, log_level="warning")
        self.server = uvicorn.Server(cfg)
        threading.Thread(target=self.server.run, daemon=True).start()
        print(f"[dashboard] open  http://localhost:{self.port}")

    def stop(self):
        self.server.should_exit = True

    def push(self, state: dict):
        if not self.loop or not self.clients or self.pending > 2:
            return
        self.pending += 1
        fut = asyncio.run_coroutine_threadsafe(self._broadcast(json.dumps(state)), self.loop)
        fut.add_done_callback(lambda _: setattr(self, "pending", self.pending - 1))

    async def _broadcast(self, text):
        for ws in list(self.clients):
            try:
                await ws.send_text(text)
            except Exception:
                self.clients.discard(ws)
