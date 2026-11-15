"""Tables and figure data from the simulations (results/sim_labels.csv, results/sim_doa.csv)."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()

    labels = list(csv.DictReader((args.results / "sim_labels.csv").open(encoding="utf-8")))
    groups = defaultdict(list)
    for row in labels:
        groups[(float(row["error_scale"]), int(row["separation_deg"]))].append(row)
    print("Speaker numbers, two people taking turns")
    lines = []
    for (scale, separation), rows in sorted(groups.items()):
        right = 100 * np.mean([r["true_number"] == r["caption_number"] for r in rows])
        switch = np.array([float(r["switch_s"]) for r in rows if r["switch_s"]])
        lines.append((scale, separation, right, np.median(switch), np.percentile(switch, 90), len(rows)))
        print(f"  error x{scale:g}, {separation:3d} deg: {right:5.1f}% of {len(rows)} captions right, "
              f"number follows the new talker after median {np.median(switch):.2f} s, 90th {np.percentile(switch, 90):.2f} s")

    doa_path = args.results / "sim_doa.csv"
    doa_lines = []
    if doa_path.exists():
        doa = list(csv.DictReader(doa_path.open(encoding="utf-8")))
        cells = defaultdict(list)
        for row in doa:
            cells[(row["method"], row["talkers"], row["t60_s"], row["snr_db"])].append(float(row["azimuth_error_deg"]))
        print("Direction finding on simulated scenes, azimuth error")
        for key, values in sorted(cells.items()):
            values = np.array(values)
            doa_lines.append(key + (np.median(values), 100 * np.mean(values <= 10), values.size))
            print(f"  {key[0]:10s} talkers {key[1]} T60 {key[2]} s SNR {key[3]:>4s} dB: median {np.median(values):5.1f} deg, "
                  f"within 10 deg {100 * np.mean(values <= 10):5.1f}%  (n = {values.size})")

    if args.figures:
        args.figures.mkdir(parents=True, exist_ok=True)
        with (args.figures / "s_labels.dat").open("w") as handle:
            handle.write("separation right1 right2 switch1 switch1p90\n")
            for separation in sorted({k[1] for k in groups}):
                one = next(l for l in lines if l[0] == 1.0 and l[1] == separation)
                two = next(l for l in lines if l[0] == 2.0 and l[1] == separation)
                handle.write(f"{separation} {one[2]:.1f} {two[2]:.1f} {one[3]:.3f} {one[4]:.3f}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
