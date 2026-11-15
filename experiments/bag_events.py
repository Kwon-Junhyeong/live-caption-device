"""Writes the message-level event tables in results/events/ from the rosbags."""
from __future__ import annotations

import argparse
import csv
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

CHUNK_MS = 128.0
MAX_SLOTS = 3


class _Cdr:
    def __init__(self, data: bytes) -> None:
        self.data, self.offset = data, 4

    def take(self, fmt: str):
        size = struct.calcsize(fmt[-1])
        self.offset += -(self.offset - 4) % size
        value = struct.unpack_from("<" + fmt, self.data, self.offset)
        self.offset += struct.calcsize("<" + fmt)
        return value

    def string(self) -> str:
        (length,) = self.take("I")
        text = self.data[self.offset:self.offset + length - 1].decode("utf-8", "replace")
        self.offset += length
        return text

    def stamp(self) -> int:
        seconds, nanoseconds = self.take("iI")
        return seconds * 1_000_000_000 + nanoseconds

    def flag(self) -> bool:
        value = bool(self.data[self.offset])
        self.offset += 1
        return value


def parse_tracks(data: bytes):
    reader = _Cdr(data)
    stamp = reader.stamp()
    reader.string()
    (count,) = reader.take("I")
    tracks = []
    for _ in range(count):
        (track_id,) = reader.take("I")
        reader.take("3d")
        azimuth, _elevation, _activity, _age = reader.take("4f")
        tracks.append((track_id, azimuth))
    return stamp, tracks


def parse_faces(data: bytes):
    reader = _Cdr(data)
    reader.stamp()
    reader.string()
    (count,) = reader.take("I")
    faces = []
    for _ in range(count):
        (track_id,) = reader.take("I")
        found = reader.flag()
        box = reader.take("4i")
        reader.take("f")
        reader.take("I")
        faces.append((track_id, found, box))
    source = reader.stamp()
    reader.take("2I")
    (detected,) = reader.take("I")
    return faces, source, detected


def parse_audio(data: bytes):
    """Stamp, sample rate, and the RMS of each track's samples in a SeparatedAudio message."""
    reader = _Cdr(data)
    stamp = reader.stamp()
    reader.string()
    sample_rate, samples = reader.take("2I")
    (count,) = reader.take("I")
    track_ids = list(reader.take(f"{count}I")) if count else []
    (total,) = reader.take("I")
    reader.offset += -(reader.offset - 4) % 4
    values = np.frombuffer(data, dtype="<f4", count=total, offset=reader.offset)
    levels = [float(np.sqrt(np.mean(values[i * samples:(i + 1) * samples] ** 2))) for i in range(len(track_ids))]
    return stamp, sample_rate, samples, track_ids, levels


def parse_caption(data: bytes):
    reader = _Cdr(data)
    reader.stamp()
    reader.string()
    (track_id,) = reader.take("I")
    text = reader.string()
    reader.take("I")
    return track_id, bool(text.strip()), reader.flag()


def replay(session: int, db3: Path, tables: dict) -> None:
    with sqlite3.connect(f"file:{db3}?mode=ro", uri=True) as connection:
        topics = dict(connection.execute("SELECT id, name FROM topics"))
        rows = connection.execute(
            "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp").fetchall()
    start = rows[0][1]
    seconds = lambda ns: f"{(ns - start) / 1e9:.6f}"
    track_side: dict[int, int] = {}
    first_match: dict[int, int] = {}
    raw_input = False
    raw_chunks = 0
    for topic_id, received, data in rows:
        topic = topics[topic_id]
        data = bytes(data)
        if topic == "/sound_source_tracks":
            stamp, tracks = parse_tracks(data)
            delay = (received - stamp) / 1e6 - CHUNK_MS
            tables["directions"].append((session, seconds(received), f"{delay:.3f}", len(tracks)))
            for track_id, azimuth in tracks:
                tables["tracks"].append((session, seconds(received), track_id, f"{azimuth:.2f}",
                                         track_side.get(track_id, 0)))
        elif topic == "/faces":
            faces, source, detected = parse_faces(data)
            tables["faces"].append((session, seconds(received), detected))
            found = sorted((f for f in faces if f[1]), key=lambda f: f[2][0] + f[2][2] / 2.0)
            matched = False
            for rank, (track_id, _found, _box) in enumerate(found):
                if track_id != 0:
                    track_side[track_id] = min(rank + 1, MAX_SLOTS)
                    matched = True
            if matched and source > 0 and source not in first_match:
                first_match[source] = received
                delay = (received - source) / 1e6 - CHUNK_MS
                tables["face_matches"].append((session, seconds(received), f"{delay:.3f}"))
        elif topic == "/separated_audio":
            stamp, rate, samples, track_ids, levels = parse_audio(data)
            raw_input = rate != 8000
            raw_chunks += raw_input
            end = stamp + int(1e9 * samples / rate)
            for track_id, level in zip(track_ids, levels):
                tables["voices"].append((session, seconds(end), track_id, f"{level:.5f}", int(raw_input)))
        elif topic == "/caption":
            track_id, has_text, final = parse_caption(data)
            if final:
                tables["captions"].append((session, seconds(received), track_id,
                                           track_side.get(track_id, 0), int(has_text), int(raw_input)))
    tables["sessions"].append((session, db3.parent.name, seconds(rows[-1][1]), raw_chunks))


HEADERS = {
    "sessions": ["session", "bag", "seconds", "raw_input_chunks"],
    "directions": ["session", "t_s", "delay_ms", "tracks"],
    "face_matches": ["session", "t_s", "delay_ms"],
    "tracks": ["session", "t_s", "track_id", "azimuth_deg", "face_number"],
    "faces": ["session", "t_s", "faces_detected"],
    "captions": ["session", "t_s", "track_id", "face_number", "has_text", "raw_input"],
    "voices": ["session", "audio_end_s", "track_id", "rms", "raw_input"],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bags", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("results/events"))
    args = parser.parse_args()
    tables = {name: [] for name in HEADERS}
    for session, bag in enumerate(sorted(args.bags), start=1):
        replay(session, next(bag.glob("*.db3")), tables)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, header in HEADERS.items():
        with (args.out / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(tables[name])
        print(f"{name}.csv: {len(tables[name])} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
