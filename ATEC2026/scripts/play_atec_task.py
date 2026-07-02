# Created by skywoodsz on 2026/02/07.

import argparse
import io
import os
import time
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play Atec Tasks (ENV only, no RL).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_recorder",
    choices=("gym", "manual"),
    default=os.getenv("ATEC_VIDEO_RECORDER", "gym"),
    help="Video recording backend. Use manual when Gym RecordVideo captures stale rendered frames.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Enable debug prints for per-step reward/time metrics.",
)
parser.add_argument(
    "--camera_mode",
    choices=("follow", "fixed", "none"),
    default=os.getenv("ATEC_CAMERA_MODE", "follow"),
    help="Viewport camera behavior for visualization and video recording.",
)
parser.add_argument(
    "--teleop_isaac_keyboard",
    action="store_true",
    default=False,
    help="Read WASD/QE velocity commands from the Isaac Sim viewport keyboard.",
)
parser.add_argument(
    "--teleop_web",
    action="store_true",
    default=False,
    help="Serve a local web teleop page with MJPEG video and WASD/QE controls.",
)
parser.add_argument(
    "--teleop_host",
    type=str,
    default=os.getenv("ATEC_TELEOP_HOST", "127.0.0.1"),
    help="Host interface for --teleop_web.",
)
parser.add_argument(
    "--teleop_port",
    type=int,
    default=int(os.getenv("ATEC_TELEOP_PORT", "8765")),
    help="Port for --teleop_web.",
)

# Isaac Sim / Kit args
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
if args_cli.teleop_isaac_keyboard and args_cli.headless:
    parser.error("--teleop_isaac_keyboard requires Isaac Sim GUI mode. Remove --headless.")
if args_cli.teleop_isaac_keyboard and args_cli.teleop_web:
    parser.error("Use only one teleop input mode: --teleop_isaac_keyboard or --teleop_web.")

# If recording video, need cameras enabled in IsaacLab/Kit
if args_cli.video or args_cli.teleop_web:
    args_cli.enable_cameras = True

# -----------------------------------------------------------------------------
# Launch Isaac Sim / Kit
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports AFTER simulation_app is created (IsaacLab pattern)
# -----------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402 (register your tasks)
from isaaclab_tasks.utils import parse_env_cfg
from rl_utils import camera_follow, set_fixed_camera
from demo.solution import AlgSolution  # noqa: E402

solution = AlgSolution()


WEB_TELEOP_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ATEC Task D Teleop</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111315;
      --panel: #1a1d20;
      --panel-2: #22262a;
      --text: #f2f4f7;
      --muted: #a7b0ba;
      --line: #343a40;
      --ok: #7dd87d;
      --warn: #f4c95d;
      --danger: #ff7a7a;
      --accent: #6ab7ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 16px;
      padding: 16px;
      min-height: 100vh;
    }
    .view {
      background: #050607;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      min-height: 360px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .view img {
      width: 100%;
      height: 100%;
      max-height: calc(100vh - 34px);
      object-fit: contain;
      display: block;
    }
    aside {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: 18px;
      line-height: 1.2;
    }
    h2 {
      margin: 0 0 8px;
      font-size: 13px;
      color: var(--muted);
      font-weight: 650;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .status {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .metric {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      min-height: 54px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .metric strong {
      font-size: 16px;
      font-weight: 650;
    }
    .keys {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 8px;
    }
    .key {
      min-height: 44px;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      font: inherit;
      font-weight: 650;
    }
    .key.active {
      border-color: var(--accent);
      background: #12324a;
    }
    .wide { grid-column: span 3; }
    .pose-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
    }
    .pose-grid label {
      display: flex;
      flex-direction: column;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
    }
    .pose-grid input {
      width: 100%;
      height: 36px;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      padding: 0 8px;
      font: inherit;
    }
    .actions {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
      margin-top: 8px;
    }
    .action {
      min-height: 36px;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      font: inherit;
      font-weight: 650;
    }
    .action.primary {
      border-color: var(--accent);
      background: #12324a;
    }
    .muted {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      margin: 8px 0 0;
    }
    .connection {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--warn);
    }
    .dot.live { background: var(--ok); }
    .dot.lost { background: var(--danger); }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      .view img { max-height: 60vh; }
    }
  </style>
</head>
<body>
  <main>
    <div class="view" aria-label="Simulation camera stream">
      <img src="/stream.mjpg" alt="Task D simulation stream">
    </div>
    <aside>
      <section>
        <h1>Task D Teleop</h1>
        <div class="connection"><span id="dot" class="dot"></span><span id="conn">Connecting</span></div>
        <p class="muted">Click this page once, then use W/S, A/D, Q/E. Hold Shift for a faster command. Space stops.</p>
      </section>
      <section>
        <h2>Command</h2>
        <div class="status">
          <div class="metric"><span>vx</span><strong id="vx">0.00</strong></div>
          <div class="metric"><span>vy</span><strong id="vy">0.00</strong></div>
          <div class="metric"><span>yaw</span><strong id="yaw">0.00</strong></div>
          <div class="metric"><span>score</span><strong id="score">0.00</strong></div>
        </div>
      </section>
      <section>
        <h2>Controls</h2>
        <div class="keys" aria-label="Keyboard controls">
          <button class="key" data-key="q">Q</button>
          <button class="key" data-key="w">W</button>
          <button class="key" data-key="e">E</button>
          <button class="key" data-key="a">A</button>
          <button class="key" data-key="s">S</button>
          <button class="key" data-key="d">D</button>
          <button class="key wide" data-key=" ">Space Stop</button>
        </div>
      </section>
      <section>
        <h2>Status</h2>
        <div class="status">
          <div class="metric"><span>step</span><strong id="step">0</strong></div>
          <div class="metric"><span>time</span><strong id="elapsed">0.00</strong></div>
          <div class="metric"><span>robot</span><strong id="robot">-</strong></div>
          <div class="metric"><span>box</span><strong id="box">-</strong></div>
        </div>
      </section>
      <section>
        <h2>Debug Pose</h2>
        <div class="pose-grid">
          <label>x<input id="pose-x" type="number" step="0.1" value="-3.0"></label>
          <label>y<input id="pose-y" type="number" step="0.1" value="0.0"></label>
          <label>z<input id="pose-z" type="number" step="0.1" value="0.8"></label>
          <label>yaw<input id="pose-yaw" type="number" step="0.1" value="0.0"></label>
        </div>
        <div class="actions">
          <button class="action primary" id="move-robot">Move Robot</button>
          <button class="action" data-preset="-3.0,0.0,0.8,0">Start</button>
          <button class="action" data-preset="-5.5,1.6,0.8,0">Behind Box</button>
          <button class="action" data-preset="-1.2,0.0,0.8,0">Near Pit</button>
          <button class="action" data-preset="1.2,0.0,0.8,0">Before Platform</button>
        </div>
      </section>
    </aside>
  </main>
  <script>
    const pressed = new Set();
    const base = { vx: 0.25, vy: 0.22, yaw: 0.35 };
    const fastScale = 1.5;
    let lastSent = "";

    function commandFromKeys() {
      const scale = pressed.has("shift") ? fastScale : 1;
      const vx = ((pressed.has("w") ? 1 : 0) - (pressed.has("s") ? 1 : 0)) * base.vx * scale;
      const vy = ((pressed.has("a") ? 1 : 0) - (pressed.has("d") ? 1 : 0)) * base.vy * scale;
      const yaw = ((pressed.has("q") ? 1 : 0) - (pressed.has("e") ? 1 : 0)) * base.yaw * scale;
      return { vx, vy, yaw };
    }

    async function sendCommand(force = false) {
      const command = commandFromKeys();
      const payload = JSON.stringify(command);
      if (!force && payload === lastSent) return;
      lastSent = payload;
      try {
        await fetch("/api/command", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payload,
          keepalive: true,
        });
      } catch (_) {}
    }

    function syncKeys() {
      document.querySelectorAll(".key").forEach((el) => {
        const key = el.dataset.key;
        el.classList.toggle("active", pressed.has(key));
      });
    }

    window.addEventListener("keydown", (event) => {
      const key = event.key.toLowerCase();
      if (["w", "a", "s", "d", "q", "e", " ", "shift"].includes(key)) {
        event.preventDefault();
        if (key === " ") pressed.clear();
        else pressed.add(key);
        syncKeys();
        sendCommand(true);
      }
    });

    window.addEventListener("keyup", (event) => {
      const key = event.key.toLowerCase();
      if (pressed.delete(key)) {
        syncKeys();
        sendCommand(true);
      }
    });

    window.addEventListener("blur", () => {
      pressed.clear();
      syncKeys();
      sendCommand(true);
    });

    window.addEventListener("beforeunload", () => {
      navigator.sendBeacon("/api/command", JSON.stringify({ vx: 0, vy: 0, yaw: 0 }));
    });

    function fmtPose(pose) {
      if (!pose) return "-";
      return `${pose[0].toFixed(2)}, ${pose[1].toFixed(2)}, ${pose[2].toFixed(2)}`;
    }

    function posePayload() {
      return {
        x: Number(document.getElementById("pose-x").value),
        y: Number(document.getElementById("pose-y").value),
        z: Number(document.getElementById("pose-z").value),
        yaw: Number(document.getElementById("pose-yaw").value),
      };
    }

    async function teleportRobot() {
      pressed.clear();
      syncKeys();
      await fetch("/api/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vx: 0, vy: 0, yaw: 0 }),
        keepalive: true,
      });
      await fetch("/api/teleport_robot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(posePayload()),
      });
    }

    document.getElementById("move-robot").addEventListener("click", teleportRobot);
    document.querySelectorAll("[data-preset]").forEach((button) => {
      button.addEventListener("click", () => {
        const [x, y, z, yaw] = button.dataset.preset.split(",");
        document.getElementById("pose-x").value = x;
        document.getElementById("pose-y").value = y;
        document.getElementById("pose-z").value = z;
        document.getElementById("pose-yaw").value = yaw;
        teleportRobot();
      });
    });

    async function pollStatus() {
      try {
        const res = await fetch("/api/status", { cache: "no-store" });
        const status = await res.json();
        document.getElementById("vx").textContent = status.command.vx.toFixed(2);
        document.getElementById("vy").textContent = status.command.vy.toFixed(2);
        document.getElementById("yaw").textContent = status.command.yaw.toFixed(2);
        document.getElementById("score").textContent = status.score.toFixed(2);
        document.getElementById("step").textContent = status.step;
        document.getElementById("elapsed").textContent = status.elapsed_time.toFixed(2);
        document.getElementById("robot").textContent = fmtPose(status.robot);
        document.getElementById("box").textContent = fmtPose(status.box);
        document.getElementById("conn").textContent = "Live";
        document.getElementById("dot").className = "dot live";
      } catch (_) {
        document.getElementById("conn").textContent = "Disconnected";
        document.getElementById("dot").className = "dot lost";
      }
    }

    setInterval(() => sendCommand(false), 80);
    setInterval(pollStatus, 200);
    pollStatus();
  </script>
</body>
</html>
"""


class IsaacKeyboardTeleop:
    def __init__(self):
        import carb.input
        import omni.appwindow

        self._carb_input = carb.input
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        self._pressed = set()
        self._vx = float(os.getenv("ATEC_TELEOP_VX", "0.25"))
        self._vy = float(os.getenv("ATEC_TELEOP_VY", "0.22"))
        self._yaw = float(os.getenv("ATEC_TELEOP_YAW", "0.35"))
        self._fast_scale = float(os.getenv("ATEC_TELEOP_FAST_SCALE", "1.5"))
        self._max_vx = float(os.getenv("ATEC_TELEOP_MAX_VX", "0.45"))
        self._max_vy = float(os.getenv("ATEC_TELEOP_MAX_VY", "0.35"))
        self._max_yaw = float(os.getenv("ATEC_TELEOP_MAX_YAW", "0.60"))

    def close(self) -> None:
        if self._keyboard_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def reset(self) -> None:
        self._pressed.clear()

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def advance(self) -> tuple[float, float, float]:
        scale = self._fast_scale if {"LEFT_SHIFT", "RIGHT_SHIFT", "SHIFT"} & self._pressed else 1.0
        vx = (("W" in self._pressed) - ("S" in self._pressed)) * self._vx * scale
        vy = (("A" in self._pressed) - ("D" in self._pressed)) * self._vy * scale
        yaw = (("Q" in self._pressed) - ("E" in self._pressed)) * self._yaw * scale
        return (
            self._clip(vx, -self._max_vx, self._max_vx),
            self._clip(vy, -self._max_vy, self._max_vy),
            self._clip(yaw, -self._max_yaw, self._max_yaw),
        )

    def _on_keyboard_event(self, event, *args, **kwargs):
        key_name = event.input.name
        if event.type == self._carb_input.KeyboardEventType.KEY_PRESS:
            if key_name in ("SPACE", "L"):
                self.reset()
            elif key_name == "P":
                vx, vy, yaw = self.advance()
                print(f"[TELEOP] vx={vx:.3f}, vy={vy:.3f}, yaw={yaw:.3f}", flush=True)
            else:
                self._pressed.add(key_name)
        elif event.type == self._carb_input.KeyboardEventType.KEY_RELEASE:
            self._pressed.discard(key_name)
        return True


class WebTeleopState:
    def __init__(self, command_timeout: float = 0.5):
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._command_timeout = command_timeout
        self._command = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
        self._last_command_time = 0.0
        self._pending_robot_teleport = None
        self._jpeg_frame = None
        self._frame_id = 0
        self._status = {
            "step": 0,
            "score": 0.0,
            "elapsed_time": 0.0,
            "command": dict(self._command),
            "robot": None,
            "box": None,
        }

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def set_command(self, vx: float, vy: float, yaw: float) -> None:
        command = {
            "vx": self._clip(float(vx), -0.45, 0.45),
            "vy": self._clip(float(vy), -0.35, 0.35),
            "yaw": self._clip(float(yaw), -0.60, 0.60),
        }
        with self._lock:
            self._command = command
            self._last_command_time = time.monotonic()
            self._status["command"] = dict(command)

    def request_robot_teleport(self, x: float, y: float, z: float, yaw: float) -> dict:
        teleport = {
            "x": self._clip(float(x), -8.0, 4.0),
            "y": self._clip(float(y), -4.0, 4.0),
            "z": self._clip(float(z), 0.4, 2.0),
            "yaw": self._clip(float(yaw), -3.14159, 3.14159),
        }
        with self._lock:
            self._pending_robot_teleport = teleport
            self._command = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
            self._last_command_time = time.monotonic()
            self._status["command"] = dict(self._command)
        return dict(teleport)

    def pop_robot_teleport(self):
        with self._lock:
            teleport = self._pending_robot_teleport
            self._pending_robot_teleport = None
        return teleport

    def get_command(self) -> tuple[float, float, float]:
        with self._lock:
            if time.monotonic() - self._last_command_time > self._command_timeout:
                self._command = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
                self._status["command"] = dict(self._command)
            command = dict(self._command)
        return command["vx"], command["vy"], command["yaw"]

    def update_status(self, step: int, score: float, elapsed_time: float, robot_pos=None, box_pos=None) -> None:
        with self._lock:
            self._status.update({
                "step": int(step),
                "score": float(score),
                "elapsed_time": float(elapsed_time),
                "command": dict(self._command),
                "robot": robot_pos,
                "box": box_pos,
            })

    def snapshot_status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def update_frame(self, frame) -> None:
        if frame is None:
            return
        frame = np.asarray(frame)
        if frame.ndim != 3:
            return
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        buffer = io.BytesIO()
        imageio.imwrite(buffer, frame, format="jpeg", quality=75)
        jpeg_frame = buffer.getvalue()
        with self._condition:
            self._jpeg_frame = jpeg_frame
            self._frame_id += 1
            self._condition.notify_all()

    def wait_frame(self, last_frame_id: int, timeout: float = 1.0):
        with self._condition:
            if self._frame_id == last_frame_id:
                self._condition.wait(timeout=timeout)
            return self._frame_id, self._jpeg_frame


def _make_web_teleop_handler(state: WebTeleopState):
    class WebTeleopHandler(BaseHTTPRequestHandler):
        server_version = "ATECTeleop/1.0"

        def log_message(self, fmt, *args):
            return

        def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                self._send_bytes(WEB_TELEOP_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if self.path == "/api/status":
                body = json.dumps(state.snapshot_status()).encode("utf-8")
                self._send_bytes(body, "application/json")
                return
            if self.path == "/stream.mjpg":
                self._stream_mjpeg()
                return
            self._send_bytes(b"not found", "text/plain", status=404)

        def do_POST(self):
            if self.path == "/api/command":
                self._handle_command()
                return
            if self.path == "/api/teleport_robot":
                self._handle_teleport_robot()
                return
            self._send_bytes(b"not found", "text/plain", status=404)

        def _read_json(self) -> dict:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            return json.loads(payload)

        def _handle_command(self):
            try:
                data = self._read_json()
                state.set_command(data.get("vx", 0.0), data.get("vy", 0.0), data.get("yaw", 0.0))
                self._send_bytes(b'{"ok": true}', "application/json")
            except Exception as exc:
                body = json.dumps({"ok": False, "error": str(exc)}).encode("utf-8")
                self._send_bytes(body, "application/json", status=400)

        def _handle_teleport_robot(self):
            try:
                data = self._read_json()
                teleport = state.request_robot_teleport(
                    data.get("x", -3.0),
                    data.get("y", 0.0),
                    data.get("z", 0.8),
                    data.get("yaw", 0.0),
                )
                body = json.dumps({"ok": True, "teleport": teleport}).encode("utf-8")
                self._send_bytes(body, "application/json")
            except Exception as exc:
                body = json.dumps({"ok": False, "error": str(exc)}).encode("utf-8")
                self._send_bytes(body, "application/json", status=400)

        def _stream_mjpeg(self):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last_frame_id = -1
            while True:
                frame_id, jpeg_frame = state.wait_frame(last_frame_id)
                if jpeg_frame is None or frame_id == last_frame_id:
                    continue
                last_frame_id = frame_id
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg_frame)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg_frame)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break

    return WebTeleopHandler


def _start_web_teleop_server(host: str, port: int, state: WebTeleopState):
    server = ThreadingHTTPServer((host, port), _make_web_teleop_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _apply_robot_teleport(robot, teleport: dict, device: str) -> None:
    yaw = float(teleport["yaw"])
    half_yaw = 0.5 * yaw
    root_pose = torch.tensor(
        [[
            float(teleport["x"]),
            float(teleport["y"]),
            float(teleport["z"]),
            float(np.cos(half_yaw)),
            0.0,
            0.0,
            float(np.sin(half_yaw)),
        ]],
        dtype=torch.float32,
        device=device,
    )
    root_velocity = torch.zeros((1, 6), dtype=torch.float32, device=device)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(root_velocity)
    if hasattr(robot, "reset"):
        robot.reset()


def _video_timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _video_output_dir(task_name: str, timestamp: str | None = None) -> str:
    parts = ["logs", "videos", task_name, "play"]
    if timestamp is not None:
        parts.append(timestamp)
    return os.path.abspath(os.path.join(*parts))


def _video_output_path(task_name: str) -> str:
    timestamp = _video_timestamp()
    filename = f"rl-video-{timestamp}.mp4"
    return os.path.join(_video_output_dir(task_name), filename)


def _apply_camera_mode(env, camera_mode: str, is_task_e: bool) -> None:
    if is_task_e:
        return
    if camera_mode == "follow":
        camera_follow(env)
    elif camera_mode == "fixed":
        set_fixed_camera(env)


def _capture_manual_frame(env):
    env.unwrapped.sim.render()
    return env.unwrapped.render(recompute=True)


def play() -> tuple[float, float]:
    if args_cli.task is None:
        raise ValueError("Please provide --task, e.g. --task ATEC-TaskA-G1")

    is_task_e = isinstance(args_cli.task, str) and args_cli.task.startswith("ATEC-TaskE")
    camera_mode = args_cli.camera_mode
    if isinstance(args_cli.task, str) and args_cli.task.startswith("ATEC-Isaac-Velocity-") and camera_mode == "follow":
        camera_mode = "none"
    # -------------------------------------------------------------------------
    # Create env (plain Gym env)
    # -------------------------------------------------------------------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )

    if args_cli.debug:
        print("[DEBUG] Creating environment...", flush=True)
    render_mode = "rgb_array" if (args_cli.video or args_cli.teleop_web) else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # Convert MARL -> single agent if needed (kept from your original script)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # -------------------------------------------------------------------------
    # Optional: video wrapper
    # -------------------------------------------------------------------------
    manual_video_path = None
    manual_video_writer = None
    manual_video_frames = 0
    if args_cli.video and args_cli.video_recorder == "gym":
        video_timestamp = _video_timestamp()
        # Put videos in ./logs/videos/play by default (edit as you like)
        video_kwargs = {
            "video_folder": _video_output_dir(args_cli.task, video_timestamp),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    elif args_cli.video and args_cli.video_recorder == "manual":
        manual_video_path = _video_output_path(args_cli.task)
        os.makedirs(os.path.dirname(manual_video_path), exist_ok=True)
        print("[INFO] Recording videos during play with manual frame capture.")
        print_dict(
            {
                "video_path": manual_video_path,
                "video_length": args_cli.video_length,
                "fps": None,
            },
            nesting=4,
        )


    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------
    if args_cli.debug:
        print("[DEBUG] Resetting environment...", flush=True)
    obs, _ = env.reset()
    if args_cli.debug:
        print(f"[DEBUG] Reset done. obs_keys={list(obs.keys())}", flush=True)
    if not is_task_e and camera_mode == "fixed":
        set_fixed_camera(env)

    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
    if manual_video_path is not None:
        fps = int(round(1.0 / dt)) if dt is not None and dt > 0 else 50
        manual_video_writer = imageio.get_writer(manual_video_path, fps=fps, quality=7, macro_block_size=1)
        _apply_camera_mode(env, camera_mode, is_task_e)
        _capture_manual_frame(env)
    timestep = 0
    robot = None
    box = None
    if hasattr(env.unwrapped, "scene"):
        try:
            robot = env.unwrapped.scene["robot"]
        except KeyError:
            robot = None
        try:
            box = env.unwrapped.scene["box"]
        except KeyError:
            box = None
    debug_taskd_pose = os.getenv("ATEC_DEBUG_TASKD_POSE", "0") == "1"
    start_root_x = None
    if robot is not None:
        start_root_x = robot.data.root_pos_w[0, 0].item()
    teleop = None
    if args_cli.teleop_isaac_keyboard:
        teleop = IsaacKeyboardTeleop()
        print("[TELEOP] Isaac keyboard control enabled.", flush=True)
        print("[TELEOP] W/S forward/back, A/D left/right, Q/E yaw, Space or L stop, P print command.", flush=True)
    web_teleop_state = None
    web_teleop_server = None
    if args_cli.teleop_web:
        web_teleop_state = WebTeleopState()
        web_teleop_server = _start_web_teleop_server(args_cli.teleop_host, args_cli.teleop_port, web_teleop_state)
        shown_host = "127.0.0.1" if args_cli.teleop_host == "0.0.0.0" else args_cli.teleop_host
        print(f"[TELEOP] Web teleop enabled: http://{shown_host}:{args_cli.teleop_port}", flush=True)
        print("[TELEOP] W/S forward/back, A/D left/right, Q/E yaw, Space stop, Shift fast.", flush=True)

    # -------------------------------------------------------------------------
    # Play loop
    # -------------------------------------------------------------------------
    total_episode_reward = 0.0
    total_elapsed_time = 0.0
    if args_cli.debug:
        print(
            f"[DEBUG] Entering play loop... headless={getattr(args_cli, 'headless', None)} "
            f"app_running={simulation_app.is_running()}",
            flush=True,
        )
    while args_cli.headless or simulation_app.is_running():
        with torch.inference_mode():
            start_time = time.time()

            # ===== Your controller goes here =====
            if args_cli.debug and timestep < 3:
                print(f"[DEBUG] step {timestep}: before predicts", flush=True)
            if web_teleop_state is not None:
                teleport = web_teleop_state.pop_robot_teleport()
                if teleport is not None and robot is not None:
                    _apply_robot_teleport(robot, teleport, env.unwrapped.device)
                    solution.set_teleop_command(0.0, 0.0, 0.0)
                    print(
                        "[TELEOP] Robot moved to "
                        f"x={teleport['x']:.2f}, y={teleport['y']:.2f}, "
                        f"z={teleport['z']:.2f}, yaw={teleport['yaw']:.2f}",
                        flush=True,
                    )
            if teleop is not None:
                solution.set_teleop_command(*teleop.advance())
            if web_teleop_state is not None:
                solution.set_teleop_command(*web_teleop_state.get_command())
            resp = solution.predicts(obs, total_episode_reward)
            if args_cli.debug and timestep < 3:
                action_preview = resp.get("action", [])
                preview_len = len(action_preview[0]) if action_preview else 0
                print(
                    f"[DEBUG] step {timestep}: after predicts action_len={preview_len} giveup={resp['giveup']}",
                    flush=True,
                )
            giveup = resp["giveup"]
            if giveup:
                break
            actions = resp["action"]
            actions = torch.tensor(actions, dtype=torch.float32, device='cuda').view(1, -1)
            if args_cli.debug and timestep < 3:
                print(f"[DEBUG] step {timestep}: before env.step action_shape={tuple(actions.shape)}", flush=True)
            obs, reward, terminated, truncated, info = env.step(actions)
            if args_cli.debug and timestep < 3:
                print(f"[DEBUG] step {timestep}: after env.step", flush=True)
            _apply_camera_mode(env, camera_mode, is_task_e)
            if args_cli.debug and timestep < 3:
                print(f"[DEBUG] step {timestep}: after camera mode", flush=True)

            frame = None
            if web_teleop_state is not None or manual_video_writer is not None:
                frame = _capture_manual_frame(env)
            if web_teleop_state is not None and frame is not None:
                web_teleop_state.update_frame(frame)
            if manual_video_writer is not None and manual_video_frames < args_cli.video_length:
                if frame is not None:
                    manual_video_writer.append_data(frame)
                    manual_video_frames += 1

            sim_dt = info.get("Step_dt", dt if dt is not None else 0.02)
            if args_cli.debug and timestep < 3:
                print(f"[DEBUG] step {timestep}: sim_dt={sim_dt}", flush=True)
            if isinstance(reward, torch.Tensor):
                total_episode_reward += reward.mean().item() / sim_dt
            else:
                total_episode_reward += float(reward) / sim_dt
            if args_cli.debug and timestep < 3:
                print(f"[DEBUG] step {timestep}: reward accumulated", flush=True)

            if isinstance(info, dict) and "Elapsed_Time" in info:
                elapsed = info["Elapsed_Time"]  # simulation time from env as primary source
                total_elapsed_time = elapsed.item() if hasattr(elapsed, "item") else float(elapsed)
            elif dt is not None:
                total_elapsed_time += dt  # wall clock time as fallback

            robot_pos = None
            box_pos = None
            if robot is not None:
                robot_pos = robot.data.root_pos_w[0, :3].detach().cpu().tolist()
            if box is not None:
                box_pos = box.data.root_pos_w[0, :3].detach().cpu().tolist()
            if web_teleop_state is not None:
                web_teleop_state.update_status(
                    timestep,
                    total_episode_reward,
                    total_elapsed_time,
                    robot_pos=robot_pos,
                    box_pos=box_pos,
                )

            if args_cli.debug:
                print(f"total_episode_reward:{total_episode_reward: .2f}")
                print(f"total_elapsed_time:{total_elapsed_time: .2f}")
                if robot is not None and timestep % 25 == 0:
                    root_x = robot.data.root_pos_w[0, 0].item()
                    delta_x = root_x - start_root_x if start_root_x is not None else 0.0
                    print(f"root_x:{root_x: .4f}, delta_x:{delta_x: .4f}")
                if debug_taskd_pose and robot is not None and box is not None and timestep % 25 == 0:
                    print(
                        "[TASKD_POSE] "
                        f"step={timestep} score={total_episode_reward:.2f} "
                        f"robot=({robot_pos[0]:.3f},{robot_pos[1]:.3f},{robot_pos[2]:.3f}) "
                        f"box=({box_pos[0]:.3f},{box_pos[1]:.3f},{box_pos[2]:.3f})",
                        flush=True,
                    )

            done = (terminated.item() or truncated.item())
            if done:
                break

            timestep += 1
            # If recording one video, exit after video_length steps
            if args_cli.video and timestep >= args_cli.video_length:
                break

            # Real-time pacing
            if args_cli.real_time and dt is not None:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    if teleop is not None:
        teleop.close()
    if web_teleop_server is not None:
        web_teleop_server.shutdown()
        web_teleop_server.server_close()
    if manual_video_writer is not None:
        manual_video_writer.close()

    env.close()

    return total_episode_reward, total_elapsed_time


if __name__ == "__main__":
    try:
        score, elapsed_time = play()
        print(f"score: {score:.2f}, elapsed_time: {elapsed_time:.2f} seconds")
    finally:
        print("Closing simulation app...")
        simulation_app.close()
