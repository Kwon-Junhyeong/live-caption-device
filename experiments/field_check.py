from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bem_matching"))
from bem import load_bem_dictionary
from doa import angular_error_deg, estimate_bem_doas
from stft import normalized_spatial_covariance, stft_multichannel

TABLE_FS = 8000
N_FFT, HOP = 512, 128
PANELS = np.array([0, 1, 2, 3], dtype=np.int64)
DEVICE_MIC_CHANNELS = [1, 2, 3, 4]


def unit_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth, elevation = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.array([np.cos(elevation) * np.cos(azimuth),
                     np.cos(elevation) * np.sin(azimuth),
                     np.sin(elevation)])


def load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio[:, DEVICE_MIC_CHANNELS]
    audio = resample_poly(audio, TABLE_FS, sample_rate, axis=0).astype(np.float32)
    return np.ascontiguousarray(audio)


def active_frames(spectra: np.ndarray, band: np.ndarray, keep: float) -> np.ndarray:
    power = np.sum(np.abs(spectra[:, band, :]) ** 2, axis=(0, 1))
    threshold = np.quantile(power, 1.0 - keep)
    return power >= threshold


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    parser.add_argument("recordings", type=Path, nargs="+")
    parser.add_argument("--band", type=float, nargs=2, default=(1000.0, 3500.0))
    parser.add_argument("--grid", type=float, default=1.0)
    parser.add_argument("--keep", type=float, default=0.5)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    low, high = args.band
    dictionary = load_bem_dictionary(args.table, PANELS, sample_rate=TABLE_FS, n_fft=N_FFT,
                                     min_hz=low, max_hz=high, grid_step_deg=args.grid)
    rows = []
    for wav in sorted(args.recordings):
        meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        truth = unit_vector(meta["azimuth_deg"], meta["elevation_deg"])
        audio = load_audio(wav)
        frequencies, spectra = stft_multichannel(audio, TABLE_FS, n_fft=N_FFT, hop=HOP)
        band = (frequencies >= low) & (frequencies <= high)
        frames = active_frames(spectra, band, args.keep)
        csm = normalized_spatial_covariance(spectra[:, band, :][:, :, frames])
        best = estimate_bem_doas(csm, dictionary, max_sources=1)[0]
        error = angular_error_deg(best.direction, truth)
        azimuth_error = (best.azimuth_deg - meta["azimuth_deg"] + 180.0) % 360.0 - 180.0
        row = {
            "recording": wav.name,
            "true_az_deg": meta["azimuth_deg"],
            "true_el_deg": meta["elevation_deg"],
            "distance_m": meta.get("distance_m"),
            "est_az_deg": round(best.azimuth_deg, 1),
            "est_el_deg": round(best.elevation_deg, 1),
            "az_error_deg": round(azimuth_error, 1),
            "el_error_deg": round(best.elevation_deg - meta["elevation_deg"], 1),
            "angular_error_deg": round(error, 1),
        }
        rows.append(row)
        print(", ".join(f"{k}={v}" for k, v in row.items()))

    errors = np.array([r["angular_error_deg"] for r in rows])
    print(f"n={len(rows)} median={np.median(errors):.1f} mean={errors.mean():.1f} max={errors.max():.1f} deg")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
