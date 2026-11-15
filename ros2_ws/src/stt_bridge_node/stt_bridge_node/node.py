from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Empty

from captioning_msgs.msg import Caption, SeparatedAudio, SoundSourceTracks


def _vendor_dir(name: str) -> str:
    try:
        share = Path(get_package_share_directory("stt_bridge_node"))
        candidate = share / name
        if candidate.is_dir():
            return str(candidate)
    except Exception:
        pass
    return str(Path(__file__).resolve().parent.parent / name)


class SttBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("stt_bridge")

        self.declare_parameter("stt_module_path", _vendor_dir("stt_module"))

        self.declare_parameter("model_size", "small")
        self.declare_parameter("language", "ko")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("compute_type", "float16")
        self.declare_parameter("frame_id", "mic_array")
        self.declare_parameter("silence_duration", 0.7)
        self.declare_parameter("min_speech_duration", 0.3)
        self.declare_parameter("partial_interval", 0.5)
        self.declare_parameter("audio_gain", 1.0)
        self.declare_parameter("silence_threshold", 0.01)

        self.declare_parameter("track_timeout_sec", 2.0)
        self.declare_parameter("max_tracks", 4)
        self.declare_parameter("publish_partial", True)

        get = lambda name: self.get_parameter(name).value
        module_path = get("stt_module_path")
        if module_path not in sys.path:
            sys.path.insert(0, module_path)
        from stt import StreamingSTT
        from stt import config as stt_config
        from stt import model as stt_model

        self._StreamingSTT = StreamingSTT
        stt_model.MODEL_SIZE = get("model_size")
        stt_config.MODEL_SIZE = get("model_size")
        stt_config.LANGUAGE = get("language")
        device = get("device")
        if device != "auto":
            compute_type = get("compute_type")
            stt_model.get_device = lambda: device
            stt_model.get_compute_type = lambda: compute_type
            stt_config.get_device = stt_model.get_device
            stt_config.get_compute_type = stt_model.get_compute_type
        self.device = device
        self.model_size = get("model_size")
        self.language = get("language")
        self.frame_id = get("frame_id")
        self.silence_duration = float(get("silence_duration"))
        self.min_speech_duration = float(get("min_speech_duration"))
        self.partial_interval = float(get("partial_interval"))
        self.audio_gain = float(get("audio_gain"))
        self.silence_threshold = float(get("silence_threshold"))
        self.track_timeout = float(get("track_timeout_sec"))
        self.max_tracks = int(get("max_tracks"))
        self.publish_partial = bool(get("publish_partial"))

        self._warm_model(stt_model)

        self.sessions: dict[int, object] = {}
        self.last_seen: dict[int, float] = {}
        self.lock = threading.Lock()

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        caption_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST, depth=50)

        self.caption_publisher = self.create_publisher(Caption, "caption", caption_qos)
        self.create_subscription(SeparatedAudio, "separated_audio", self._on_audio, sensor_qos)
        self.create_subscription(SoundSourceTracks, "sound_source_tracks",
                                 self._on_tracks, sensor_qos)
        self.create_subscription(Empty, "caption_reset", self._on_reset, 10)
        self.create_timer(1.0, self._reap)

        self.get_logger().info(
            f"STT module {module_path}, model {self.model_size}, device {self.device}, "
            f"language {self.language}, gain {self.audio_gain}, silence threshold {self.silence_threshold}")

    def _warm_model(self, stt_model) -> None:
        started = self.get_clock().now().nanoseconds / 1e9
        try:
            stt_model.get_model()
        except Exception as error:
            self.get_logger().warn(f"model warm-up failed: {error}")
            return
        elapsed = self.get_clock().now().nanoseconds / 1e9 - started
        self.get_logger().info(
            f"Whisper {self.model_size} warmed up in {elapsed:.1f}s ({self.device})")

    def _session(self, track_id: int):
        session = self.sessions.get(track_id)
        if session is not None:
            return session
        if len(self.sessions) >= self.max_tracks:
            self.get_logger().warn(
                f"track limit {self.max_tracks} reached, ignoring track {track_id}",
                throttle_duration_sec=10.0)
            return None

        def on_result(result, track_id=track_id) -> None:
            self._publish(track_id, result)

        session = self._StreamingSTT(on_result=on_result,
                                     sample_rate=16000,
                                     partial_interval=self.partial_interval,
                                     silence_duration=self.silence_duration,
                                     silence_threshold=self.silence_threshold,
                                     min_speech_duration=self.min_speech_duration)
        session.start()
        self.sessions[track_id] = session
        self.get_logger().info(f"track {track_id} STT session started ({len(self.sessions)} open)")
        return session

    def _publish(self, track_id: int, result) -> None:
        if result.type == "error":
            self.get_logger().error(f"track {track_id} STT error: {result.error}")
            return
        if result.type == "partial" and not self.publish_partial:
            return
        message = Caption()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.track_id = int(track_id)
        message.text = result.text or ""
        message.utterance_id = int(result.utterance_id)
        message.is_final = bool(result.is_final)
        self.caption_publisher.publish(message)

    def _on_audio(self, message: SeparatedAudio) -> None:
        if message.n_samples == 0:
            return
        if not message.track_ids:

            quiet = np.zeros(int(message.n_samples), dtype=np.float32)
            with self.lock:
                for session in list(self.sessions.values()):
                    try:
                        session.push_audio(quiet, sample_rate=int(message.sample_rate))
                    except Exception:
                        pass
            return
        samples = np.asarray(message.samples, dtype=np.float32)
        n = int(message.n_samples)
        with self.lock:
            for index, track_id in enumerate(message.track_ids):
                chunk = samples[index * n:(index + 1) * n]
                if chunk.size != n or not np.isfinite(chunk).all():
                    continue
                if self.audio_gain != 1.0:
                    chunk = np.clip(chunk * self.audio_gain, -1.0, 1.0)
                session = self._session(int(track_id))
                if session is None:
                    continue
                self.last_seen[int(track_id)] = self.get_clock().now().nanoseconds / 1e9
                try:
                    session.push_audio(chunk, sample_rate=int(message.sample_rate))
                except Exception as error:
                    self.get_logger().error(f"track {track_id} push failed: {error}")

    def _on_tracks(self, message: SoundSourceTracks) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        with self.lock:
            for track in message.tracks:
                self.last_seen[int(track.track_id)] = now

    def _reap(self) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        with self.lock:
            stale = [t for t, seen in self.last_seen.items()
                     if now - seen > self.track_timeout and t in self.sessions]
            for track_id in stale:
                session = self.sessions.pop(track_id)
                self.last_seen.pop(track_id, None)
                try:
                    session.flush()
                    session.stop()
                except Exception as error:
                    self.get_logger().warn(f"track {track_id} while closing: {error}")
                self.get_logger().info(f"track {track_id} session closed ({len(self.sessions)} open)")

    def _on_reset(self, _message: Empty) -> None:
        with self.lock:
            for track_id, session in list(self.sessions.items()):
                try:
                    session.stop()
                except Exception:
                    pass
            self.sessions.clear()
        self.get_logger().info("caption reset, all STT sessions closed")

    def destroy_node(self) -> bool:
        with self.lock:
            for track_id, session in list(self.sessions.items()):
                try:
                    session.flush()
                    session.stop()
                except Exception:
                    pass
            self.sessions.clear()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SttBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
