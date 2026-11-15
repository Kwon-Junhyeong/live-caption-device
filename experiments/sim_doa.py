"""Direction finding on simulated scenes of this housing: steering vectors with the housing's scattering against
the same search with free-field steering vectors, and SRP-PHAT with free-field steering vectors."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bem_matching"))
from stft import normalized_spatial_covariance, stft_multichannel  # noqa: E402
from sim_room import (C, FS, Housing, diffuse_noise, impulse_response, load_speech,  # noqa: E402
                      place_talker, random_room)

N_FFT, HOP = 512, 128
GRID_DEG = 2
BAND = (300.0, 3800.0)


def search_grid(housing: Housing) -> np.ndarray:
    az = np.arange(0, 360, GRID_DEG)
    el = np.arange(-40, 71, GRID_DEG)
    return (az[:, None] * housing.n_el + (el[None, :] + 90)).ravel()


def steering_sets(housing: Housing, grid: np.ndarray, freqs: np.ndarray) -> dict[str, np.ndarray]:
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    scattered = housing.pressure[grid][:, band, :]
    units = housing.grid[grid]
    delays = units @ housing.mics.T / C                                   # [direction, mic], s
    free = np.exp(2j * np.pi * freqs[band][None, :, None] * delays[:, None, :])
    norm = lambda a: a / np.linalg.norm(a, axis=2, keepdims=True)
    return {"scattering": norm(scattered).astype(np.complex64), "free_field": norm(free).astype(np.complex64)}, band


def outer(steering: np.ndarray) -> np.ndarray:
    """conj(a_i) a_j for every direction, flattened so that a search is one matrix-vector product."""
    n = steering.shape[0]
    return (np.conj(steering)[:, :, :, None] * steering[:, :, None, :]).reshape(n, -1)


def scores(products: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    return np.real(products @ covariance.reshape(-1).astype(np.complex64))


def peaks(score: np.ndarray, units: np.ndarray, count: int, min_sep_deg: float = 20.0) -> list[int]:
    limit = np.cos(np.radians(min_sep_deg))
    picked: list[int] = []
    for index in np.argsort(score)[::-1]:
        if len(picked) == count:
            break
        if picked and np.any(units[picked] @ units[index] > limit):
            continue
        picked.append(int(index))
    return picked


def errors(estimates: np.ndarray, truths: np.ndarray) -> list[tuple[float, float]]:
    """Angle between estimate and truth, and azimuth difference, after matching estimates to talkers."""
    angle = np.degrees(np.arccos(np.clip(estimates @ truths.T, -1, 1)))
    rows, cols = linear_sum_assignment(angle)
    azimuth = lambda v: np.degrees(np.arctan2(v[1], v[0]))
    return [(angle[r, c], abs((azimuth(estimates[r]) - azimuth(truths[c]) + 180) % 360 - 180))
            for r, c in zip(rows, cols)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    parser.add_argument("speech", type=Path, nargs="+")
    parser.add_argument("--scenes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--out", type=Path, default=Path("results/sim_doa.csv"))
    args = parser.parse_args()

    housing = Housing(args.table)
    clips = [load_speech(p) for p in args.speech]
    grid = search_grid(housing)
    units = housing.grid[grid]
    freqs = np.fft.rfftfreq(N_FFT, 1 / FS)
    steering, band = steering_sets(housing, grid, freqs)
    products = {name: outer(value) for name, value in steering.items()}
    rng = np.random.default_rng(args.seed)
    rows = []
    for t60 in (0.3, 0.7):
        for snr in (20.0, 5.0):
            for talkers in (1, 2):
                for scene in range(args.scenes):
                    room = random_room(rng, t60)
                    azimuths = [rng.uniform(30, 150)]
                    while talkers == 2 and len(azimuths) < 2:
                        candidate = rng.uniform(30, 150)
                        if abs(candidate - azimuths[0]) >= 20:
                            azimuths.append(candidate)
                    length = 4 * FS
                    mixture = np.zeros((length, 4))
                    truths = []
                    for azimuth in azimuths:
                        position, unit = place_talker(rng, room, azimuth)
                        clip = clips[rng.integers(len(clips))]
                        start = rng.integers(0, max(1, clip.size - length))
                        signal = np.zeros(length)
                        piece = clip[start:start + length]
                        signal[:piece.size] = piece
                        ir = impulse_response(housing, room, position)
                        mixture += fftconvolve(signal[:, None], ir, axes=0)[:length]
                        truths.append(unit)
                    noise = diffuse_noise(housing, length, rng)
                    gain = np.sqrt(np.mean(mixture ** 2) / np.mean(noise ** 2) / 10 ** (snr / 10))
                    mixture += gain * noise
                    _, spectra = stft_multichannel(mixture, FS, N_FFT, HOP)
                    spectra = spectra[:, band, :]
                    power = np.sum(np.abs(spectra) ** 2, axis=(0, 1))
                    spectra = spectra[:, :, power >= np.quantile(power, 0.5)]
                    covariance = normalized_spatial_covariance(spectra)
                    phat = spectra / np.maximum(np.abs(spectra), 1e-12)
                    phat_cov = np.einsum("cft,dft->fcd", phat, np.conj(phat), optimize=True) / phat.shape[-1]
                    truths = np.asarray(truths)
                    for method, product, cov in (("scattering", products["scattering"], covariance),
                                                 ("free_field", products["free_field"], covariance),
                                                 ("srp_phat", products["free_field"], phat_cov)):
                        picked = peaks(scores(product, cov), units, talkers)
                        for error, azimuth in errors(units[picked], truths):
                            rows.append((method, t60, snr, talkers, scene, f"{error:.2f}", f"{azimuth:.2f}"))
                print(f"T60 {t60}, SNR {snr}, done", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "t60_s", "snr_db", "talkers", "scene", "error_deg", "azimuth_error_deg"])
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
