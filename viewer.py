from __future__ import annotations

import argparse
import ctypes
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from imgui_bundle import hello_imgui, imgui
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BGR,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_ATTACHMENT0,
    GL_COMPILE_STATUS,
    GL_DEPTH_TEST,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_LINEAR,
    GL_LINK_STATUS,
    GL_RGB,
    GL_STATIC_DRAW,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLE_STRIP,
    GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glBindBuffer,
    glBindFramebuffer,
    glBindTexture,
    glBindVertexArray,
    glBufferData,
    glCheckFramebufferStatus,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteBuffers,
    glDeleteFramebuffers,
    glDeleteProgram,
    glDeleteShader,
    glDeleteTextures,
    glDeleteVertexArrays,
    glDisable,
    glDrawArrays,
    glEnableVertexAttribArray,
    glFramebufferTexture2D,
    glGenBuffers,
    glGenFramebuffers,
    glGenTextures,
    glGenVertexArrays,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetUniformLocation,
    glLinkProgram,
    glPixelStorei,
    glShaderSource,
    glTexImage2D,
    glTexParameteri,
    glTexSubImage2D,
    glUniform1f,
    glUniform1i,
    glUniform2f,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-file panorama + PTZ viewer with imgui docking. Supports ROS 2 raw Image, video file, or a live THETA camera via GStreamer/thetauvcsrc."
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        default=None,
        help="Optional path to an equirectangular video file. Ignored if --ros-topic is used.",
    )
    parser.add_argument(
        "--ros-topic",
        dest="ros_topic",
        default=None,
        help="ROS 2 raw sensor_msgs/msg/Image topic, e.g. /camera/image_decoded",
    )
    parser.add_argument(
        "--camera-mode",
        default="4K",
        choices=["2K", "4K"],
        help="THETA camera mode for the default live-camera input path (thetauvcsrc).",
    )
    parser.add_argument(
        "--theta-serial",
        default=None,
        help="Optional THETA serial number to select when multiple cameras are connected.",
    )
    parser.add_argument(
        "--gst-pipeline",
        dest="gst_pipeline",
        default=None,
        help="Optional full GStreamer pipeline string. If omitted, a thetauvcsrc-based pipeline is used.",
    )
    parser.add_argument(
        "--gst-pull-timeout-ms",
        type=int,
        default=200,
        help="Timeout in milliseconds when waiting for a frame from appsink in live camera mode.",
    )
    parser.add_argument("--width", type=int, default=None, help="Optional target panorama width.")
    parser.add_argument("--height", type=int, default=None, help="Optional target panorama height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Kept for CLI compatibility; not used by the default thetauvcsrc camera pipeline.")
    parser.add_argument("--loop-video", action="store_true", help="Loop the video file when it reaches the end.")
    parser.add_argument("--reliable", action="store_true", help="Use RELIABLE QoS for ROS subscriptions.")
    return parser.parse_args()


# =========================
# Shared state
# =========================
@dataclass
class PreviewImage:
    frame_id: int = -1
    ts: float = 0.0
    pano_bgr_small: Optional[np.ndarray] = None


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._preview = PreviewImage()
        self._paused = False
        self._capture_fps = 0.0
        self._status = "starting..."
        self._last_error: Optional[str] = None
        self._start_ts = time.perf_counter()
        self._source_desc = ""
        self._frame_shape: Optional[tuple[int, int, int]] = None

    def set_source_desc(self, desc: str) -> None:
        with self._lock:
            self._source_desc = desc

    def toggle_paused(self) -> None:
        with self._lock:
            self._paused = not self._paused

    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def put_pano_preview(self, frame_id: int, pano_bgr: np.ndarray, ts: float) -> None:
        pano_bgr = np.ascontiguousarray(pano_bgr)
        with self._lock:
            self._preview.frame_id = frame_id
            self._preview.ts = ts
            self._preview.pano_bgr_small = pano_bgr
            self._frame_shape = tuple(pano_bgr.shape)

    def get_latest_preview(self) -> PreviewImage:
        with self._lock:
            return self._preview

    def set_status(self, value: str) -> None:
        with self._lock:
            self._status = value

    def set_error(self, value: Optional[str]) -> None:
        with self._lock:
            self._last_error = value

    def set_capture_fps(self, fps: float) -> None:
        with self._lock:
            self._capture_fps = fps if self._capture_fps == 0.0 else (0.9 * self._capture_fps + 0.1 * fps)

    def ui_snapshot(self) -> dict:
        with self._lock:
            return {
                "uptime_s": time.perf_counter() - self._start_ts,
                "status": self._status,
                "last_error": self._last_error,
                "paused": self._paused,
                "capture_fps": self._capture_fps,
                "source_desc": self._source_desc,
                "frame_shape": self._frame_shape,
            }


# =========================
# Input sources
# =========================
class ThetaGStreamerSource:
    def __init__(
        self,
        *,
        camera_mode: str = "4K",
        theta_serial: Optional[str] = None,
        gst_pipeline: Optional[str] = None,
        pull_timeout_ms: int = 200,
    ) -> None:
        self.camera_mode = str(camera_mode)
        self.theta_serial = theta_serial
        self.gst_pipeline = gst_pipeline
        self.pull_timeout_ms = int(pull_timeout_ms)

        self.Gst = None
        self.GstVideo = None
        self.pipeline = None
        self.appsink = None
        self.bus = None
        self.pipeline_desc = ""

    def _build_default_pipeline(self) -> str:
        serial_part = f' serial="{self.theta_serial}"' if self.theta_serial else ""
        return (
            f"thetauvcsrc mode={self.camera_mode}{serial_part} "
            f"! queue "
            f"! h264parse "
            f"! decodebin "
            f"! queue "
            f"! videoconvert "
            f"! video/x-raw,format=BGR "
            f"! appsink name=theta_appsink emit-signals=false sync=false max-buffers=1 drop=true"
        )

    def _check_bus(self) -> None:
        if self.bus is None or self.Gst is None:
            return

        message_types = self.Gst.MessageType.ERROR | self.Gst.MessageType.EOS
        while True:
            msg = self.bus.timed_pop_filtered(0, message_types)
            if msg is None:
                break
            if msg.type == self.Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                extra = f" | debug={debug}" if debug else ""
                raise RuntimeError(f"GStreamer error: {err}{extra}")
            if msg.type == self.Gst.MessageType.EOS:
                raise RuntimeError("GStreamer camera stream reached EOS.")

    def open(self) -> None:
        try:
            import gi

            gi.require_version("Gst", "1.0")
            gi.require_version("GstVideo", "1.0")
            from gi.repository import Gst, GstVideo
        except Exception as exc:  # pragma: no cover - import depends on system packages
            raise RuntimeError(
                "GStreamer Python bindings are not available. On Ubuntu install at least: "
                "python3-gi python3-gst-1.0 gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 "
                "gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav."
            ) from exc

        self.Gst = Gst
        self.GstVideo = GstVideo

        Gst.init(None)
        self.pipeline_desc = self.gst_pipeline or self._build_default_pipeline()

        try:
            self.pipeline = Gst.parse_launch(self.pipeline_desc)
        except Exception as exc:
            raise RuntimeError(f"Failed to create GStreamer pipeline: {exc}\nPipeline: {self.pipeline_desc}") from exc

        self.appsink = self.pipeline.get_by_name("theta_appsink")
        if self.appsink is None:
            raise RuntimeError(
                "GStreamer pipeline does not contain an appsink named 'theta_appsink'. "
                "Provide one in --gst-pipeline or use the default thetauvcsrc pipeline."
            )

        self.bus = self.pipeline.get_bus()
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self._check_bus()
            raise RuntimeError(f"Failed to set GStreamer pipeline to PLAYING.\nPipeline: {self.pipeline_desc}")

        state_ret, state, pending = self.pipeline.get_state(5 * Gst.SECOND)
        if state_ret == Gst.StateChangeReturn.FAILURE:
            self._check_bus()
            raise RuntimeError(f"GStreamer pipeline failed during startup.\nPipeline: {self.pipeline_desc}")

    def _sample_to_bgr(self, sample) -> np.ndarray:
        Gst = self.Gst
        GstVideo = self.GstVideo
        if Gst is None or GstVideo is None:
            raise RuntimeError("GStreamer is not initialized.")

        caps = sample.get_caps()
        if caps is None or caps.get_size() == 0:
            raise RuntimeError("Appsink sample did not contain valid caps.")

        structure = caps.get_structure(0)
        fmt = str(structure.get_value("format"))
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))

        video_info = GstVideo.VideoInfo()
        if not video_info.from_caps(caps):
            raise RuntimeError(f"Could not parse video caps: {caps.to_string()}")
        stride = int(video_info.stride[0])

        buffer = sample.get_buffer()
        if buffer is None:
            raise RuntimeError("Appsink sample did not contain a GstBuffer.")

        ok, map_info = buffer.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("Could not map GstBuffer for reading.")

        try:
            raw = np.frombuffer(map_info.data, dtype=np.uint8)

            if fmt == "BGR":
                row_bytes = width * 3
                frame = raw.reshape((height, stride))[:, :row_bytes].reshape((height, width, 3))
                return np.ascontiguousarray(frame)

            if fmt == "BGRx":
                row_bytes = width * 4
                frame = raw.reshape((height, stride))[:, :row_bytes].reshape((height, width, 4))
                return np.ascontiguousarray(frame[:, :, :3])

            if fmt == "RGB":
                row_bytes = width * 3
                frame = raw.reshape((height, stride))[:, :row_bytes].reshape((height, width, 3))
                return np.ascontiguousarray(frame[:, :, ::-1])

            if fmt == "RGBx":
                row_bytes = width * 4
                frame = raw.reshape((height, stride))[:, :row_bytes].reshape((height, width, 4))
                return np.ascontiguousarray(frame[:, :, :3][:, :, ::-1])

            raise RuntimeError(f"Unsupported appsink format: {fmt}")
        finally:
            buffer.unmap(map_info)

    def read_frame(self) -> Optional[np.ndarray]:
        if self.appsink is None or self.Gst is None:
            return None

        self._check_bus()
        timeout_ns = int(self.pull_timeout_ms) * self.Gst.MSECOND
        if hasattr(self.appsink, "try_pull_sample"):
            sample = self.appsink.try_pull_sample(timeout_ns)
        else:
            sample = self.appsink.emit("try-pull-sample", timeout_ns)
        if sample is None:
            self._check_bus()
            return None
        return self._sample_to_bgr(sample)

    def close(self) -> None:
        if self.pipeline is not None and self.Gst is not None:
            try:
                self.pipeline.set_state(self.Gst.State.NULL)
            except Exception:
                pass
        self.appsink = None
        self.bus = None
        self.pipeline = None


class OpenCVVideoFileSource:
    def __init__(self, path: str, loop: bool = True) -> None:
        self.path = str(path)
        self.loop = bool(loop)
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.path}")

    def read_frame(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if ok and frame is not None:
            return frame
        if not self.loop:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class ROSImageSource:
    def __init__(
        self,
        *,
        topic: str,
        node_name: str = "panorama_ptz_ros_sub",
        queue_size: int = 1,
        wait_timeout_sec: float = 1.0,
        reliable: bool = False,
    ) -> None:
        self.topic = topic
        self.node_name = node_name
        self.queue_size = int(queue_size)
        self.wait_timeout_sec = float(wait_timeout_sec)
        self.reliable = bool(reliable)

        self._node = None
        self._spin_thread: Optional[threading.Thread] = None
        self._owns_rclpy = False
        self._cond = threading.Condition()
        self._latest: Optional[np.ndarray] = None
        self._seq = 0
        self._last_read_seq = 0
        self._last_error: Optional[str] = None
        self._closed = False

    @staticmethod
    def _ros_image_to_bgr(msg) -> np.ndarray:
        h = int(msg.height)
        w = int(msg.width)
        step = int(msg.step)
        enc = str(msg.encoding).lower()

        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image size: {w}x{h}")

        buf = np.frombuffer(msg.data, dtype=np.uint8)

        if enc == "bgr8":
            expected_row = w * 3
            arr = buf.reshape((h, step))[:, :expected_row]
            return np.ascontiguousarray(arr.reshape((h, w, 3)))

        if enc == "rgb8":
            expected_row = w * 3
            arr = buf.reshape((h, step))[:, :expected_row]
            rgb = np.ascontiguousarray(arr.reshape((h, w, 3)))
            return rgb[:, :, ::-1]

        if enc == "bgra8":
            expected_row = w * 4
            arr = buf.reshape((h, step))[:, :expected_row]
            bgra = np.ascontiguousarray(arr.reshape((h, w, 4)))
            return bgra[:, :, :3]

        if enc == "rgba8":
            expected_row = w * 4
            arr = buf.reshape((h, step))[:, :expected_row]
            rgba = np.ascontiguousarray(arr.reshape((h, w, 4)))
            rgb = rgba[:, :, :3]
            return rgb[:, :, ::-1]

        if enc in ("mono8", "8uc1"):
            expected_row = w
            arr = buf.reshape((h, step))[:, :expected_row]
            gray = np.ascontiguousarray(arr.reshape((h, w)))
            return np.repeat(gray[:, :, None], 3, axis=2)

        raise ValueError(f"Unsupported ROS image encoding: {msg.encoding}")

    def _on_image(self, frame: np.ndarray) -> None:
        with self._cond:
            self._latest = frame
            self._last_error = None
            self._seq += 1
            self._cond.notify_all()

    def _on_error(self, message: str) -> None:
        with self._cond:
            self._last_error = message
            self._cond.notify_all()

    def open(self) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True

        reliability = ReliabilityPolicy.RELIABLE if self.reliable else ReliabilityPolicy.BEST_EFFORT

        outer = self

        class _ROSSubscriber(Node):
            def __init__(self) -> None:
                super().__init__(outer.node_name)
                qos = QoSProfile(
                    history=HistoryPolicy.KEEP_LAST,
                    depth=max(1, int(outer.queue_size)),
                    reliability=reliability,
                )
                self._sub = self.create_subscription(Image, outer.topic, self._cb, qos)

            def _cb(self, msg: Image) -> None:
                try:
                    frame = outer._ros_image_to_bgr(msg)
                    outer._on_image(frame)
                except Exception as exc:  # pragma: no cover - runtime path
                    outer._on_error(f"ROS image conversion failed: {exc}")

        self._node = _ROSSubscriber()
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True, name="ROSImageSpin")
        self._spin_thread.start()

    def read_frame(self) -> Optional[np.ndarray]:
        deadline = time.monotonic() + self.wait_timeout_sec
        with self._cond:
            start_seq = self._last_read_seq
            while not self._closed and self._seq == start_seq and self._last_error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._cond.wait(timeout=remaining)

            if self._last_error is not None or self._latest is None:
                return None

            self._last_read_seq = self._seq
            return self._latest.copy()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

        try:
            if self._node is not None:
                self._node.destroy_node()
        except Exception:
            pass

        try:
            import rclpy

            if self._owns_rclpy and rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)

        self._node = None
        self._spin_thread = None


# =========================
# Capture worker
# =========================
class CaptureWorker(threading.Thread):
    def __init__(self, source, state: SharedState, target_size: Optional[tuple[int, int]]) -> None:
        super().__init__(name="CaptureWorker", daemon=True)
        self.source = source
        self.state = state
        self.target_size = target_size
        self.stop_event = threading.Event()
        self.frame_id = 0

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        last_ts: Optional[float] = None
        try:
            self.source.open()
            self.state.set_status("running")
            self.state.set_error(None)

            while not self.stop_event.is_set():
                if self.state.is_paused():
                    time.sleep(0.01)
                    continue

                frame = self.source.read_frame()
                if frame is None:
                    time.sleep(0.005)
                    continue

                if self.target_size is not None:
                    target_w, target_h = self.target_size
                    if frame.shape[1] != target_w or frame.shape[0] != target_h:
                        interpolation = cv2.INTER_AREA if frame.shape[1] > target_w else cv2.INTER_LINEAR
                        frame = cv2.resize(frame, (target_w, target_h), interpolation=interpolation)

                now = time.perf_counter()
                self.state.put_pano_preview(self.frame_id, frame, now)

                if last_ts is not None:
                    dt = max(now - last_ts, 1e-6)
                    self.state.set_capture_fps(1.0 / dt)
                last_ts = now
                self.frame_id += 1
        except Exception as exc:
            self.state.set_status("error")
            self.state.set_error(str(exc))
        finally:
            try:
                self.source.close()
            except Exception:
                pass


# =========================
# OpenGL texture wrapper
# =========================
@dataclass
class GLTexture:
    tex_id: int = 0
    w: int = 0
    h: int = 0

    def ensure(self) -> None:
        if self.tex_id != 0:
            return
        self.tex_id = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def allocate(self, w: int, h: int) -> None:
        self.ensure()
        if w == self.w and h == self.h:
            return
        self.w, self.h = w, h
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_BGR, GL_UNSIGNED_BYTE, None)

    def upload_bgr(self, img_bgr: np.ndarray) -> None:
        h, w = img_bgr.shape[:2]
        self.allocate(w, h)
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, img_bgr)

    def destroy(self) -> None:
        if self.tex_id != 0:
            try:
                glDeleteTextures([self.tex_id])
            except Exception:
                pass
            self.tex_id = 0
            self.w = 0
            self.h = 0


# =========================
# PTZ shader renderer
# =========================
VERT_SRC = r"""
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aUv;
out vec2 vUv;
void main() {
    vUv = aUv;
    gl_Position = vec4(aPos.xy, 0.0, 1.0);
}
"""

FRAG_SRC = r"""
#version 330 core
in vec2 vUv;
out vec4 FragColor;

uniform sampler2D uPano;
uniform float uYaw;
uniform float uPitch;
uniform float uHfov;
uniform vec2  uOutSize;

const float PI = 3.14159265358979323846;

mat3 rotX(float a) {
    float c = cos(a), s = sin(a);
    return mat3(
        1.0, 0.0, 0.0,
        0.0,  c, -s,
        0.0,  s,  c
    );
}
mat3 rotY(float a) {
    float c = cos(a), s = sin(a);
    return mat3(
         c, 0.0,  s,
        0.0, 1.0, 0.0,
        -s, 0.0,  c
    );
}

void main() {
    vec2 ndc = vUv * 2.0 - 1.0;
    float aspect = uOutSize.x / uOutSize.y;
    float tanHalfH = tan(uHfov * 0.5);
    float tanHalfV = tanHalfH / aspect;

    vec3 dir = normalize(vec3(ndc.x * tanHalfH, ndc.y * tanHalfV, 1.0));
    dir = rotY(uYaw) * rotX(-uPitch) * dir;

    float lon = atan(dir.x, dir.z);
    float lat = asin(clamp(dir.y, -1.0, 1.0));

    float u = lon / (2.0 * PI) + 0.5;
    float v = 0.5 - lat / PI;

    u = fract(u);
    v = clamp(v, 0.0, 1.0);

    vec3 rgb = texture(uPano, vec2(u, v)).rgb;
    FragColor = vec4(rgb, 1.0);
}
"""


def _compile_shader(src: str, shader_type: int) -> int:
    sh = glCreateShader(shader_type)
    glShaderSource(sh, src)
    glCompileShader(sh)
    ok = glGetShaderiv(sh, GL_COMPILE_STATUS)
    if not ok:
        info = glGetShaderInfoLog(sh).decode(errors="ignore")
        raise RuntimeError(f"Shader compile failed:\n{info}")
    return sh


def _link_program(vs: int, fs: int) -> int:
    from OpenGL.GL import glAttachShader

    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    ok = glGetProgramiv(program, GL_LINK_STATUS)
    if not ok:
        info = glGetProgramInfoLog(program).decode(errors="ignore")
        raise RuntimeError(f"Program link failed:\n{info}")
    return program


@dataclass
class PTZState:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    hfov_deg: float = 90.0


class PTZRenderer:
    def __init__(self) -> None:
        self._init = False
        self.program = 0
        self.vao = 0
        self.vbo = 0
        self.fbo = 0
        self.out_tex = 0
        self.out_w = 0
        self.out_h = 0
        self.u_pano = -1
        self.u_yaw = -1
        self.u_pitch = -1
        self.u_hfov = -1
        self.u_out_size = -1

    def ensure_initialized(self) -> None:
        if self._init:
            return

        vs = _compile_shader(VERT_SRC, GL_VERTEX_SHADER)
        fs = _compile_shader(FRAG_SRC, GL_FRAGMENT_SHADER)
        self.program = _link_program(vs, fs)
        glDeleteShader(vs)
        glDeleteShader(fs)

        self.u_pano = glGetUniformLocation(self.program, "uPano")
        self.u_yaw = glGetUniformLocation(self.program, "uYaw")
        self.u_pitch = glGetUniformLocation(self.program, "uPitch")
        self.u_hfov = glGetUniformLocation(self.program, "uHfov")
        self.u_out_size = glGetUniformLocation(self.program, "uOutSize")

        quad = np.array(
            [
                -1.0, -1.0, 0.0, 0.0,
                 1.0, -1.0, 1.0, 0.0,
                -1.0,  1.0, 0.0, 1.0,
                 1.0,  1.0, 1.0, 1.0,
            ],
            dtype=np.float32,
        )

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)

        stride = 4 * 4
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * 4))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        self._init = True

    def ensure_fbo(self, w: int, h: int) -> None:
        self.ensure_initialized()
        w, h = int(w), int(h)
        if w <= 0 or h <= 0:
            return
        if self.out_tex != 0 and self.fbo != 0 and w == self.out_w and h == self.out_h:
            return

        self.out_w, self.out_h = w, h

        if self.out_tex == 0:
            self.out_tex = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self.out_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

        if self.fbo == 0:
            self.fbo = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.out_tex, 0)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"FBO not complete (status={status})")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def render(self, pano_tex_id: int, state: PTZState, out_size: tuple[int, int]) -> int:
        self.ensure_fbo(out_size[0], out_size[1])
        glDisable(GL_DEPTH_TEST)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.out_w, self.out_h)

        glUseProgram(self.program)
        glUniform1i(self.u_pano, 0)
        glUniform1f(self.u_yaw, math.radians(state.yaw_deg))
        glUniform1f(self.u_pitch, math.radians(state.pitch_deg))
        glUniform1f(self.u_hfov, math.radians(state.hfov_deg))
        glUniform2f(self.u_out_size, float(self.out_w), float(self.out_h))

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, int(pano_tex_id))

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)

        glUseProgram(0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        return self.out_tex

    def destroy(self) -> None:
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])
            self.vbo = 0
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
            self.vao = 0
        if self.program:
            glDeleteProgram(self.program)
            self.program = 0
        if self.fbo:
            glDeleteFramebuffers(1, [self.fbo])
            self.fbo = 0
        if self.out_tex:
            glDeleteTextures([self.out_tex])
            self.out_tex = 0
        self._init = False


# =========================
# Geometry / thumbnail helpers
# =========================
def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _frustum_outline_uv_for_thumbnail(ptz: PTZState, aspect: float, samples_per_edge: int = 32) -> tuple[np.ndarray, np.ndarray]:
    yaw = math.radians(-ptz.yaw_deg)
    pitch = math.radians(-ptz.pitch_deg)
    hfov = math.radians(ptz.hfov_deg)

    tan_half_h = math.tan(hfov * 0.5)
    tan_half_v = tan_half_h / max(1e-6, aspect)
    rotation = (_rot_y(yaw) @ _rot_x(-pitch)).astype(np.float32)

    n = max(12, int(samples_per_edge))
    xs = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, n, dtype=np.float32)

    top = np.stack([xs, np.full(n, 1.0, np.float32)], axis=1)
    right = np.stack([np.full(n, 1.0, np.float32), ys[::-1]], axis=1)
    bottom = np.stack([xs[::-1], np.full(n, -1.0, np.float32)], axis=1)
    left = np.stack([np.full(n, -1.0, np.float32), ys], axis=1)
    ndc = np.concatenate([top, right, bottom, left], axis=0)

    rays = np.stack(
        [ndc[:, 0] * tan_half_h, ndc[:, 1] * tan_half_v, np.ones(ndc.shape[0], dtype=np.float32)],
        axis=1,
    )
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-8)

    direction = (rotation @ rays.T).T
    lon = np.arctan2(direction[:, 0], direction[:, 2])
    lat = np.arcsin(np.clip(direction[:, 1], -1.0, 1.0))

    u = (lon / (2.0 * math.pi)) + 0.5
    v = 0.5 - (lat / math.pi)
    return u.astype(np.float32), np.clip(v, 0.0, 1.0).astype(np.float32)


def _unwrap_u(u: np.ndarray) -> np.ndarray:
    uu = u.astype(np.float32).copy()
    for i in range(1, len(uu)):
        du = uu[i] - uu[i - 1]
        if du > 0.5:
            uu[i:] -= 1.0
        elif du < -0.5:
            uu[i:] += 1.0
    return uu


# =========================
# Main GUI
# =========================
class ViewerGui:
    def __init__(self, state: SharedState) -> None:
        self.state = state
        self.pano_tex = GLTexture()
        self._last_uploaded_pano_id = -1

        self.ptz = PTZRenderer()
        self.ptz_state = PTZState(yaw_deg=0.0, pitch_deg=0.0, hfov_deg=90.0)
        self._ptz_dirty = True
        self._ptz_last_input_frame = -1
        self._ptz_out_tex_id = 0
        self._ptz_render_scale = 1.0

    def _imgui_image(self, tex_id: int, disp_w: int, disp_h: int, *, flip_v: bool = False) -> None:
        tex_ref = imgui.ImTextureRef(int(tex_id))
        if flip_v:
            imgui.image(tex_ref, (disp_w, disp_h), uv0=(0, 1), uv1=(1, 0))
        else:
            imgui.image(tex_ref, (disp_w, disp_h), uv0=(0, 0), uv1=(1, 1))

    def _recenter_ptz_to_mouse(self, disp_w: int, disp_h: int) -> None:
        mouse_pos = imgui.get_mouse_pos()
        rect_min = imgui.get_item_rect_min()
        rx = float(mouse_pos.x - rect_min.x)
        ry = float(mouse_pos.y - rect_min.y)

        u = max(0.0, min(1.0, rx / max(1.0, float(disp_w))))
        v = max(0.0, min(1.0, ry / max(1.0, float(disp_h))))

        ndc_x = 2.0 * u - 1.0
        ndc_y = 1.0 - 2.0 * v

        hfov = math.radians(self.ptz_state.hfov_deg)
        aspect = float(disp_w) / max(1.0, float(disp_h))
        tan_half_h = math.tan(hfov * 0.5)
        tan_half_v = tan_half_h / max(1e-6, aspect)

        ray = np.array([ndc_x * tan_half_h, ndc_y * tan_half_v, 1.0], dtype=np.float32)
        ray /= max(1e-8, float(np.linalg.norm(ray)))

        yaw = math.radians(self.ptz_state.yaw_deg)
        pitch = math.radians(self.ptz_state.pitch_deg)
        rotation = (_rot_y(yaw) @ _rot_x(-pitch)).astype(np.float32)
        direction = rotation @ ray

        lon = math.atan2(float(direction[0]), float(direction[2]))
        lat = math.asin(max(-1.0, min(1.0, float(direction[1]))))

        self.ptz_state.yaw_deg = float((math.degrees(lon) + 180.0) % 360.0 - 180.0)
        self.ptz_state.pitch_deg = float(max(-89.0, min(89.0, math.degrees(lat))))
        self._ptz_dirty = True

    def _handle_ptz_interaction(self, disp_w: int, disp_h: int) -> None:
        io = imgui.get_io()
        if not imgui.is_item_hovered():
            return

        if io.mouse_wheel != 0.0:
            step = 8.0 if io.key_shift else 4.0
            self.ptz_state.hfov_deg -= io.mouse_wheel * step
            self.ptz_state.hfov_deg = float(max(20.0, min(120.0, self.ptz_state.hfov_deg)))
            self._ptz_dirty = True
            io.mouse_wheel = 0.0

        if imgui.is_mouse_double_clicked(imgui.MouseButton_.left):
            self._recenter_ptz_to_mouse(disp_w, disp_h)
            return

        if io.mouse_down[0]:
            dx = float(io.mouse_delta.x)
            dy = float(io.mouse_delta.y)
            if dx != 0.0 or dy != 0.0:
                hfov_deg = float(self.ptz_state.hfov_deg)
                aspect = float(disp_w) / max(1.0, float(disp_h))
                vfov_deg = math.degrees(2.0 * math.atan(math.tan(math.radians(hfov_deg) * 0.5) / max(1e-6, aspect)))
                self.ptz_state.yaw_deg += (dx / max(1.0, disp_w)) * hfov_deg
                self.ptz_state.pitch_deg += (-dy / max(1.0, disp_h)) * vfov_deg
                self.ptz_state.yaw_deg = float((self.ptz_state.yaw_deg + 180.0) % 360.0 - 180.0)
                self.ptz_state.pitch_deg = float(max(-89.0, min(89.0, self.ptz_state.pitch_deg)))
                self._ptz_dirty = True

    def _ensure_pano_uploaded(self) -> Optional[PreviewImage]:
        preview = self.state.get_latest_preview()
        if preview.pano_bgr_small is None:
            return None
        if preview.frame_id != self._last_uploaded_pano_id:
            self.pano_tex.upload_bgr(preview.pano_bgr_small)
            self._last_uploaded_pano_id = preview.frame_id
        return preview

    def pano_window_gui(self) -> None:
        snap = self.state.ui_snapshot()
        preview = self._ensure_pano_uploaded()

        imgui.text(f"Status: {snap['status']}")
        imgui.same_line()
        imgui.text(f"Uptime: {snap['uptime_s']:.1f}s")
        imgui.text(f"Source: {snap['source_desc']}")
        imgui.text(f"Capture FPS(est): {snap['capture_fps']:.1f}")
        if snap["frame_shape"] is not None:
            h, w, _ = snap["frame_shape"]
            imgui.text(f"Current frame: {w} x {h}")
        if snap["last_error"]:
            imgui.text_colored((1.0, 0.35, 0.35, 1.0), f"Error: {snap['last_error']}")

        imgui.separator()
        if imgui.button("Resume" if snap["paused"] else "Pause"):
            self.state.toggle_paused()

        imgui.separator()
        if preview is None:
            imgui.text_disabled("Waiting for panorama preview...")
            return

        avail_w = max(200.0, float(imgui.get_content_region_avail().x))
        h, w = preview.pano_bgr_small.shape[:2]
        disp_w = int(avail_w)
        disp_h = int(disp_w * (h / w))

        imgui.text("Panorama")
        self._imgui_image(self.pano_tex.tex_id, disp_w, disp_h, flip_v=False)

    def ptz_window_gui(self) -> None:
        preview = self._ensure_pano_uploaded()
        if preview is None:
            imgui.text_disabled("Waiting for panorama preview...")
            return

        if self.pano_tex.tex_id == 0 or self.pano_tex.w <= 0 or self.pano_tex.h <= 0:
            imgui.text_disabled("PTZ: pano texture not ready yet")
            return

        imgui.text("PTZ: drag LMB pan/tilt • wheel zoom • double-click recenter")
        imgui.text_disabled("Double-click recenters at cursor. Thumbnail shows the current PTZ footprint.")

        if imgui.button("Reset PTZ"):
            self.ptz_state = PTZState(yaw_deg=0.0, pitch_deg=0.0, hfov_deg=90.0)
            self._ptz_dirty = True

        spacing = float(imgui.get_style().item_spacing.x)
        avail_x = float(imgui.get_content_region_avail().x - 30 * spacing)
        item_w = max(60.0, (avail_x - 3 * 3 * spacing) / 4.0)

        imgui.push_item_width(item_w)
        changed, value = imgui.slider_float("Yaw##ptz_yaw", float(self.ptz_state.yaw_deg), -180.0, 180.0)
        if changed:
            self.ptz_state.yaw_deg = float(value)
            self._ptz_dirty = True

        imgui.same_line(0.0, 3 * spacing)
        changed, value = imgui.slider_float("Pitch##ptz_pitch", float(self.ptz_state.pitch_deg), -89.0, 89.0)
        if changed:
            self.ptz_state.pitch_deg = float(value)
            self._ptz_dirty = True

        imgui.same_line(0.0, 3 * spacing)
        changed, value = imgui.slider_float("HFOV##ptz_hfov", float(self.ptz_state.hfov_deg), 20.0, 120.0)
        if changed:
            self.ptz_state.hfov_deg = float(value)
            self._ptz_dirty = True

        imgui.same_line(0.0, 3 * spacing)
        changed, value = imgui.slider_float("Scale##ptz_scale", float(self._ptz_render_scale), 0.25, 1.0)
        if changed:
            self._ptz_render_scale = float(value)
            self._ptz_dirty = True
        imgui.pop_item_width()

        if preview.frame_id != self._ptz_last_input_frame:
            self._ptz_last_input_frame = preview.frame_id
            self._ptz_dirty = True

        out_w = max(1, int(round(self.pano_tex.w * self._ptz_render_scale)))
        out_h = max(1, int(round(self.pano_tex.h * self._ptz_render_scale)))
        if self._ptz_dirty:
            self._ptz_out_tex_id = self.ptz.render(self.pano_tex.tex_id, self.ptz_state, (out_w, out_h))
            self._ptz_dirty = False

        avail = imgui.get_content_region_avail()
        max_w = max(200.0, float(avail.x))
        max_h = max(120.0, float(avail.y))

        pano_aspect = self.pano_tex.w / max(1.0, float(self.pano_tex.h))
        disp_w = min(max_w, max_h * pano_aspect)
        disp_h = disp_w / max(1e-6, pano_aspect)
        disp_w_i = int(disp_w)
        disp_h_i = int(disp_h)

        self._imgui_image(self._ptz_out_tex_id, disp_w_i, disp_h_i, flip_v=True)
        self._handle_ptz_interaction(disp_w_i, disp_h_i)
        self._draw_pano_thumbnail_with_roi_poly(disp_w_i, disp_h_i)

    def _draw_pano_thumbnail_with_roi_poly(self, disp_w: int, disp_h: int) -> None:
        if self.pano_tex.tex_id == 0:
            return

        img_min = imgui.get_item_rect_min()
        img_max = imgui.get_item_rect_max()
        img_w = float(img_max.x - img_min.x)
        img_h = float(img_max.y - img_min.y)
        if img_w <= 2 or img_h <= 2:
            return

        margin = 10.0
        max_size = 170.0
        pano_aspect = self.pano_tex.w / max(1.0, float(self.pano_tex.h))
        if pano_aspect >= 1.0:
            thumb_w = max_size
            thumb_h = max_size / pano_aspect
        else:
            thumb_h = max_size
            thumb_w = max_size * pano_aspect

        max_w = max(30.0, img_w - 2 * margin)
        max_h = max(30.0, img_h - 2 * margin)
        scale = min(max_w / thumb_w, max_h / thumb_h, 1.0)
        thumb_w *= scale
        thumb_h *= scale

        x0 = float(img_max.x - margin - thumb_w)
        y0 = float(img_max.y - margin - thumb_h)
        x1 = x0 + thumb_w
        y1 = y0 + thumb_h

        draw_list = imgui.get_window_draw_list()
        white = imgui.get_color_u32(imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
        tex = imgui.ImTextureRef(int(self.pano_tex.tex_id))
        draw_list.add_image_quad(
            tex,
            imgui.ImVec2(x0, y0),
            imgui.ImVec2(x0, y1),
            imgui.ImVec2(x1, y1),
            imgui.ImVec2(x1, y0),
            imgui.ImVec2(0.0, 0.0),
            imgui.ImVec2(0.0, 1.0),
            imgui.ImVec2(1.0, 1.0),
            imgui.ImVec2(1.0, 0.0),
            white,
        )
        draw_list.add_quad(
            imgui.ImVec2(x0, y0),
            imgui.ImVec2(x0, y1),
            imgui.ImVec2(x1, y1),
            imgui.ImVec2(x1, y0),
            white,
            2.0,
        )

        ptz_aspect = float(disp_w) / max(1.0, float(disp_h))
        u, v = _frustum_outline_uv_for_thumbnail(self.ptz_state, ptz_aspect, samples_per_edge=32)
        uu = _unwrap_u(u)

        def to_xy(ui: float, vi: float) -> tuple[float, float]:
            return x0 + float((ui % 1.0)) * thumb_w, y0 + float(vi) * thumb_h

        for i in range(1, len(uu)):
            if int(math.floor(float(uu[i - 1]))) != int(math.floor(float(uu[i]))):
                continue
            xa, ya = to_xy(float(uu[i - 1]), float(v[i - 1]))
            xb, yb = to_xy(float(uu[i]), float(v[i]))
            draw_list.add_line(imgui.ImVec2(xa, ya), imgui.ImVec2(xb, yb), white, 2.0)

        if int(math.floor(float(uu[-1]))) == int(math.floor(float(uu[0]))):
            xa, ya = to_xy(float(uu[-1]), float(v[-1]))
            xb, yb = to_xy(float(uu[0]), float(v[0]))
            draw_list.add_line(imgui.ImVec2(xa, ya), imgui.ImVec2(xb, yb), white, 2.0)

    def before_exit(self) -> None:
        self.pano_tex.destroy()
        self.ptz.destroy()


# =========================
# Docking layout and run loop
# =========================
def _create_default_docking_splits() -> list[hello_imgui.DockingSplit]:
    split = hello_imgui.DockingSplit()
    split.initial_dock = "MainDockSpace"
    split.new_dock = "RightDockSpace"
    split.direction = imgui.Dir.right
    split.ratio = 0.50
    return [split]


def _create_dockable_windows(gui: ViewerGui) -> list[hello_imgui.DockableWindow]:
    windows: list[hello_imgui.DockableWindow] = []

    pano = hello_imgui.DockableWindow()
    pano.label = "Panorama"
    pano.dock_space_name = "MainDockSpace"
    pano.gui_function = gui.pano_window_gui
    pano.include_in_view_menu = True
    pano.remember_is_visible = True
    windows.append(pano)

    ptz = hello_imgui.DockableWindow()
    ptz.label = "PTZ"
    ptz.dock_space_name = "RightDockSpace"
    ptz.gui_function = gui.ptz_window_gui
    ptz.include_in_view_menu = True
    ptz.remember_is_visible = True
    ptz.imgui_window_flags = imgui.WindowFlags_.no_scroll_with_mouse | imgui.WindowFlags_.no_scrollbar
    windows.append(ptz)

    return windows


def run_gui(state: SharedState) -> None:
    gui = ViewerGui(state)

    runner = hello_imgui.RunnerParams()
    runner.app_window_params.window_title = "360 Panorama PTZ Viewer"
    runner.app_window_params.window_geometry.size = (1280, 900)

    runner.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
    runner.imgui_window_params.enable_viewports = False
    runner.imgui_window_params.show_menu_bar = True
    runner.imgui_window_params.show_menu_view = True
    runner.imgui_window_params.show_menu_app = False
    runner.fps_idling.enable_idling = False

    docking = hello_imgui.DockingParams()
    docking.docking_splits = _create_default_docking_splits()
    docking.dockable_windows = _create_dockable_windows(gui)
    docking.layout_condition = hello_imgui.DockingLayoutCondition.application_start
    runner.docking_params = docking
    runner.callbacks.before_exit = gui.before_exit

    hello_imgui.run(runner)


def build_source(args: argparse.Namespace):
    if args.ros_topic:
        desc = f"ROS 2 topic: {args.ros_topic}"
        return ROSImageSource(topic=args.ros_topic, reliable=args.reliable), desc

    if args.video_path:
        desc = f"Video file: {Path(args.video_path).name}"
        return OpenCVVideoFileSource(args.video_path, loop=args.loop_video), desc

    source = ThetaGStreamerSource(
        camera_mode=args.camera_mode,
        theta_serial=args.theta_serial,
        gst_pipeline=args.gst_pipeline,
        pull_timeout_ms=args.gst_pull_timeout_ms,
    )
    if args.gst_pipeline:
        desc = "Live camera: custom GStreamer pipeline"
    else:
        desc = f"Live camera: thetauvcsrc ({args.camera_mode})"
        if args.theta_serial:
            desc += f" serial={args.theta_serial}"
    return source, desc


def main() -> int:
    args = parse_args()

    state = SharedState()
    source, desc = build_source(args)
    state.set_source_desc(desc)

    target_size = None
    if args.width is not None and args.height is not None:
        target_size = (int(args.width), int(args.height))

    worker = CaptureWorker(source, state, target_size=target_size)
    worker.start()

    try:
        run_gui(state)
    finally:
        worker.stop()
        worker.join(timeout=2.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
