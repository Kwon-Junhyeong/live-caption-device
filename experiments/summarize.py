"""Computes every number in the Checks section from results/events/ and writes the figure data."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

GAIN, SILENCE = 3.0, 0.05      # the recognition node's audio gain and silence threshold in the launch file
SENTENCE_GAP = 0.128 + 0.7     # a voice pause of 0.7 s ends a sentence in the recognition session


def load(folder: Path, name: str) -> list[dict]:
    with (folder / f"{name}.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def by_session(rows: list[dict], column: str) -> dict[int, np.ndarray]:
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["session"])].append(float(row[column]))
    return {s: np.array(v) for s, v in groups.items()}


def histogram(values: np.ndarray, low: float, high: float, width: float, path: Path) -> None:
    edges = np.arange(low, high + width, width)
    counts, _ = np.histogram(values, edges)
    share = 100 * counts / values.size
    nonzero = np.flatnonzero(counts)
    a, b = max(nonzero[0] - 1, 0), min(nonzero[-1] + 2, counts.size)
    with path.open("w") as handle:
        handle.write("x p\n")
        for x, p in zip(edges[a:b], share[a:b]):
            handle.write(f"{x:.1f} {p:.4f}\n")
        handle.write(f"{edges[b]:.1f} 0\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("events", type=Path, nargs="?", default=Path("results/events"))
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()

    sessions = load(args.events, "sessions")
    directions = load(args.events, "directions")
    matches = load(args.events, "face_matches")
    tracks = load(args.events, "tracks")
    faces = load(args.events, "faces")
    captions = load(args.events, "captions")

    times = by_session(directions, "t_s")
    delays = by_session(directions, "delay_ms")
    face_delays = by_session(matches, "delay_ms")
    intervals = {s: np.diff(t) * 1000 for s, t in times.items()}
    all_intervals = np.concatenate(list(intervals.values()))
    all_delays = np.concatenate(list(delays.values()))
    all_face = np.concatenate(list(face_delays.values()))

    print("Timing")
    print(f"  direction messages {all_delays.size}")
    print("  interval mean per session " + ", ".join(f"{intervals[s].mean():.1f}" for s in sorted(intervals)))
    print(f"  interval median {np.median(all_intervals):.1f}, 99th percentile {np.percentile(all_intervals, 99):.1f} ms")
    print(f"  delay to direction median {np.median(all_delays):.1f}, 95th percentile {np.percentile(all_delays, 95):.1f} ms")
    print("  per session median " + ", ".join(f"{np.median(delays[s]):.1f}" for s in sorted(delays)))
    tenth = {s: (d[: max(1, d.size // 10)].mean(), d[-max(1, d.size // 10):].mean()) for s, d in delays.items()}
    print("  first and last tenth " + ", ".join(f"{a:.1f}/{b:.1f}" for a, b in tenth.values()))
    print(f"  face matches {all_face.size}, median {np.median(all_face):.1f}, 95th percentile {np.percentile(all_face, 95):.1f} ms")
    print("  per session median " + ", ".join(f"{np.median(face_delays[s]):.1f}" for s in sorted(face_delays)))
    s, i = max(((s, int(np.argmax(v))) for s, v in intervals.items()), key=lambda p: intervals[p[0]][p[1]])
    gap_start, gap_end = times[s][i], times[s][i + 1]
    after = delays[s][i + 1:]
    late = int(np.argmax(after < 420))
    back = int(np.argmax(after < 250))
    print(f"  longest interval {intervals[s][i]:.0f} ms in session {s}, {gap_start:.2f} to {gap_end:.2f} s; "
          f"next {late} messages {after[:late].max():.0f} to {after[:late].min():.0f} ms; "
          f"{after[back]:.0f} ms at {times[s][i + 1 + back] - gap_end:.2f} s after the gap")

    print("End of speech to caption")
    voices = load(args.events, "voices")
    voiced = defaultdict(list)
    for v in voices:
        if float(v["rms"]) * GAIN > SILENCE:
            voiced[(v["session"], v["track_id"], v["raw_input"])].append(float(v["audio_end_s"]))
    caption_rows = []
    for c in captions:
        if c["has_text"] != "1":
            continue
        t = float(c["t_s"])
        chunks = sorted(voiced[(c["session"], c["track_id"], c["raw_input"])])
        runs, run_end = [], None
        for e in chunks:
            if run_end is not None and e - run_end > SENTENCE_GAP:
                runs.append(run_end)
            run_end = e
        if run_end is not None:
            runs.append(run_end)
        ends = [e for e in runs if e <= t]
        if not ends:
            continue
        busy = {v["track_id"] for v in voices if v["session"] == c["session"] and v["raw_input"] == "0"
                and t - 10 <= float(v["audio_end_s"]) <= t}
        caption_rows.append((int(c["session"]), int(c["track_id"]), int(c["raw_input"]), t - max(ends), len(busy)))
    caption_delay = np.array([r[3] for r in caption_rows])
    one = np.array([r[3] for r in caption_rows if r[4] <= 1])
    many = np.array([r[3] for r in caption_rows if r[4] >= 2])
    print(f"  {caption_delay.size} captions, median {np.median(caption_delay):.2f} s, "
          f"within 2.5 s {100 * np.mean(caption_delay <= 2.5):.0f}%")
    print(f"  at most one track with voice in the last 10 s: {one.size} captions, median {np.median(one):.2f} s, "
          f"90th percentile {np.percentile(one, 90):.2f} s")
    print(f"  two or more tracks: {many.size} captions, median {np.median(many):.2f} s; over 5 s: "
          f"{int(np.sum(caption_delay > 5))} captions, {int(np.sum(many > 5))} of them with two or more tracks")
    with (args.events.parent / "caption_delay.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["session", "track_id", "raw_input", "delay_s", "tracks_with_voice_in_last_10_s"])
        writer.writerows([(a, b, c, f"{d:.3f}", e) for a, b, c, d, e in caption_rows])

    print("Speaker numbers")
    chunks = defaultdict(int)
    for row in directions:
        chunks[int(row["session"])] += int(row["tracks"]) > 0
    matched = {k: v.size for k, v in face_delays.items()}
    most = defaultdict(int)
    for row in faces:
        most[int(row["session"])] = max(most[int(row["session"])], int(row["faces_detected"]))
    rows = []
    for meta in sessions:
        k = int(meta["session"])
        mine = [c for c in captions if int(c["session"]) == k]
        separated = [c for c in mine if c["has_text"] == "1" and c["raw_input"] == "0"]
        rows.append({
            "session": k, "bag": meta["bag"], "seconds": round(float(meta["seconds"]), 1),
            "direction_messages": times[k].size, "most_faces_in_a_frame": most[k],
            "chunks_with_a_track": chunks[k], "chunks_matched_to_a_face": matched.get(k, 0),
            "raw_input_chunks": int(meta["raw_input_chunks"]),
            "final_results": len(mine),
            "without_text": sum(c["has_text"] == "0" for c in mine),
            "raw_input_with_text": sum(c["has_text"] == "1" and c["raw_input"] == "1" for c in mine),
            "separated_with_text": len(separated),
            "separated_with_a_number": sum(c["face_number"] != "0" for c in separated),
            "interval_mean_ms": round(float(intervals[k].mean()), 1),
            "delay_median_ms": round(float(np.median(delays[k])), 1),
            "face_delay_median_ms": round(float(np.median(face_delays[k])), 1) if k in face_delays else "",
        })
    for row in rows:
        print("  " + ", ".join(f"{key}={value}" for key, value in row.items()))
    with (args.events.parent / "sessions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if args.figures:
        out = args.figures
        out.mkdir(parents=True, exist_ok=True)
        histogram(all_intervals, 0, 260, 5, out / "t_interval.dat")
        histogram(all_delays, 140, 420, 5, out / "t_delay.dat")
        histogram(all_face, 140, 420, 5, out / "t_face.dat")
        with (out / "t_caption.dat").open("w") as handle:
            handle.write("x n\n")
            edges = np.arange(0, 6.25, 0.25)
            counts, _ = np.histogram(np.clip(caption_delay, 0, 6.0 - 1e-9), edges)
            for x, n in zip(edges[:-1], counts):
                handle.write(f"{x:.2f} {n}\n")
            handle.write(f"{edges[-1]:.2f} 0\n")
        with (out / "t_stall.dat").open("w") as handle:
            handle.write("t d\n")
            for t, d in zip(times[s], delays[s]):
                if gap_start - 5 <= t <= gap_end + 5:
                    handle.write(f"{t:.3f} {d:.1f}\n")
        with (out / "t_gap.dat").open("w") as handle:
            handle.write(f"start end\n{gap_start:.3f} {gap_end:.3f}\n")
        turn = 3
        segments = defaultdict(list)
        with (out / "u_unmatched.dat").open("w") as handle:
            handle.write("t az\n")
            for row in tracks:
                if int(row["session"]) != turn:
                    continue
                t, az, number = float(row["t_s"]), float(row["azimuth_deg"]), int(row["face_number"])
                if number == 0:
                    handle.write(f"{t:.3f} {az:.2f}\n")
                else:
                    segments[int(row["track_id"])].append((t, az, number))
        with (out / "u_tracks.dat").open("w") as handle:
            handle.write("t az\n")
            for points in segments.values():
                last = None
                for t, az, _n in points:
                    if last is not None and t - last > 0.5:
                        handle.write("\n")
                    handle.write(f"{t:.3f} {az:.2f}\n")
                    last = t
                handle.write("\n")
        with (out / "u_labels.dat").open("w") as handle:
            handle.write("track start end az number\n")
            for track, points in segments.items():
                numbers = [p[2] for p in points]
                handle.write(f"{track} {points[0][0]:.2f} {points[-1][0]:.2f} {np.median([p[1] for p in points]):.1f} "
                             f"{max(set(numbers), key=numbers.count)}\n")
        with (out / "u_faces.dat").open("w") as handle:
            handle.write("t n\n")
            mine = [row for row in faces if int(row["session"]) == turn]
            previous = None
            for row in mine:
                if row["faces_detected"] != previous:
                    handle.write(f"{row['t_s']} {row['faces_detected']}\n")
                    previous = row["faces_detected"]
            handle.write(f"{mine[-1]['t_s']} {mine[-1]['faces_detected']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
