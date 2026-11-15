"""Two people taking turns, run through this repository's tracker, track-to-face matching and numbering.

Each 128 ms chunk the network is replaced by one active slot pointing at the talker, with a direction error drawn
so that its median is the network's median error on single-speaker dev clips (5.4 degrees). The tracker
(model/tracker.py) and the camera node's projection (camera_node/node.py) are imported from the repository;
the gates, the Hungarian match within 20 degrees and the left-to-right numbering follow the node code line by line.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import types
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent / "ros2_ws" / "src"
sys.path.insert(0, str(ROOT / "doa_separation_node" / "doa_separation_node" / "model"))
sys.path.insert(0, str(ROOT / "camera_node"))
from tracker import SourceTracker, TrackerConfig  # noqa: E402


def _stub_ros() -> None:
    """Lets camera_node.node import without ROS 2; only its projection is used."""
    for name in ("rclpy", "rclpy.node", "rclpy.qos", "cv2", "sensor_msgs", "sensor_msgs.msg",
                 "captioning_msgs", "captioning_msgs.msg"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["rclpy.node"].Node = object
    for attr in ("QoSDurabilityPolicy", "QoSHistoryPolicy", "QoSProfile", "QoSReliabilityPolicy"):
        setattr(sys.modules["rclpy.qos"], attr, object)
    sys.modules["sensor_msgs.msg"].CompressedImage = object
    for attr in ("Face", "Faces", "SoundSourceTracks"):
        setattr(sys.modules["captioning_msgs.msg"], attr, object)


_stub_ros()
from camera_node.node import CameraNode  # noqa: E402

CHUNK = 0.128
CAMERA_PERIOD = 0.1
FX, CX, CY = 640.0, 640.0, 400.0          # assumed intrinsics: 1280 x 800 color image, 90 degree field of view
TALKER_DISTANCE, MOUTH, DEVICE = 2.0, 1.2, 0.85
MEDIAN_ERROR_DEG = 5.4
CAPTION_DELAY = 2.0                        # median end of voice to recorded caption in the device sessions


def camera() -> CameraNode:
    node = CameraNode.__new__(CameraNode)
    node.nominal_distance, node.cam_offset = 3.0, np.array([0.0, 0.0, 0.07])
    node.yaw_cos, node.yaw_sin, node.lateral_sign = 1.0, 0.0, 1.0
    node.fx = node.fy = FX
    node.cx, node.cy = CX, CY
    return node


def talker_position(azimuth_deg: float) -> np.ndarray:
    a = math.radians(azimuth_deg)
    return np.array([TALKER_DISTANCE * math.cos(a), TALKER_DISTANCE * math.sin(a), MOUTH - DEVICE])


def face_center(node: CameraNode, position: np.ndarray) -> tuple[float, float]:
    point = position - node.cam_offset
    camera_xyz = np.array([node.lateral_sign * point[0], -point[2], point[1]])
    return (node.fx * camera_xyz[0] / camera_xyz[2] + node.cx, node.fy * camera_xyz[1] / camera_xyz[2] + node.cy)


def noisy(unit: np.ndarray, sigma_deg: float, rng: np.random.Generator) -> np.ndarray:
    tangent = rng.normal(scale=math.radians(sigma_deg), size=3)
    tangent -= unit * (tangent @ unit)
    v = unit + tangent
    return v / np.linalg.norm(v)


def run(separation: float, error_scale: float, rng: np.random.Generator, turns: int = 20):
    node = camera()
    azimuths = (90 + separation / 2, 90 - separation / 2)            # talker 0 on the left of the image
    positions = [talker_position(a) for a in azimuths]
    units = [p / np.linalg.norm(p) for p in positions]
    faces = [face_center(node, p) for p in positions]
    order = sorted(range(2), key=lambda k: faces[k][0])
    seat = {k: order.index(k) + 1 for k in range(2)}
    sigma = error_scale * MEDIAN_ERROR_DEG / math.sqrt(2 * math.log(2))
    radius = node.fx * math.tan(math.radians(20.0))

    schedule, t = [], 0.5
    for turn in range(turns):
        length = rng.uniform(1.5, 4.0)
        schedule.append((turn % 2, t, t + length))
        t += length + rng.uniform(0.8, 1.5)
    end = t + 3.0

    tracker = SourceTracker(TrackerConfig(activity_threshold=0.3, gate_deg=35.0, smooth_halflife_sec=0.6,
                                          confirm_chunks=2, coast_chunks=32))
    track_side: dict[int, int] = {}
    side_log: list[tuple[float, int, int]] = []
    published: list = []
    hold, next_camera, time = 0, 0.0, 0.0
    carried: dict[int, list] = {i: [] for i in range(len(schedule))}
    active: list[tuple[float, int]] = []
    while time < end:
        speaking = next((i for i, (_, a, b) in enumerate(schedule) if a <= time < b), None)
        directions = rng.normal(size=(3, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        activity = rng.uniform(0.0, 0.1, size=3)
        slot = None
        if speaking is not None:
            slot = int(rng.integers(3)) if not carried[speaking] else carried[speaking][-1][0]
            who = schedule[speaking][0]
            directions[slot] = noisy(units[who], sigma, rng)
            activity[slot] = rng.uniform(0.8, 1.0)
        tracks = tracker.update(directions, activity, dt=CHUNK)
        if speaking is not None:
            hold = 8
        elif hold > 0:
            hold -= 1
        else:
            tracks = []
        tracks = [k for k in tracks if 30 <= k.azimuth_deg <= 150 and -10 <= k.elevation_deg <= 35]
        tracks = sorted(tracks, key=lambda k: -k.activity)[:2]
        published = [(k.track_id, k.direction.copy(), k.activity) for k in tracks]
        if speaking is not None:
            mapping = tracker.slot_to_track()
            carried[speaking].append((slot, mapping.get(slot)))
        while next_camera <= time:
            if published:
                cost = np.full((len(published), 2), 1e6)
                for i, (_, direction, _a) in enumerate(published):
                    point = node._project(direction)
                    if point is None:
                        continue
                    for j, (x, y) in enumerate(faces):
                        distance = math.hypot(x - point[0], y - point[1])
                        if distance <= radius:
                            cost[i, j] = distance
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows, cols):
                    if cost[r, c] < 1e6:
                        number = order.index(c) + 1
                        if track_side.get(published[r][0]) != number:
                            side_log.append((time, published[r][0], number))
                        track_side[published[r][0]] = number
            next_camera += CAMERA_PERIOD
        loudest = max(published, key=lambda p: p[2])[0] if published else None
        active.append((time, track_side.get(loudest, 0) if loudest is not None else 0))
        time += CHUNK

    results = []
    for i, (who, start, stop) in enumerate(schedule):
        ids = [track for _slot, track in carried[i] if track is not None]
        if not ids:
            results.append((who, 0, None))
            continue
        track = max(set(ids), key=ids.count)
        caption_time = stop + CAPTION_DELAY
        number = 0
        for when, logged, value in side_log:
            if logged == track and when <= caption_time:
                number = value
        results.append((who, number, _switch(active, start, seat[who]) if i else None))
    return seat, results


def _switch(active, start, target):
    for t, number in active:
        if t >= start and number == target:
            return t - start
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--out", type=Path, default=Path("results/sim_labels.csv"))
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for error_scale in (1.0, 2.0):
        for separation in (10, 20, 30, 35, 40, 50, 60, 90):
            for run_index in range(args.runs):
                seat, results = run(separation, error_scale, rng)
                for k, (who, number, switch) in enumerate(results):
                    rows.append((error_scale, separation, run_index, k, who, seat[who], number,
                                 "" if switch is None else f"{switch:.3f}"))
            print(f"error x{error_scale}, separation {separation} done", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["error_scale", "separation_deg", "run", "utterance", "talker", "true_number",
                         "caption_number", "switch_s"])
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
