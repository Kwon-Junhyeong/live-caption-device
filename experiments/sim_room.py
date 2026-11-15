"""Renders four-microphone scenes for this housing: image-source room plus the BEM surface pressure for every arrival."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import fftconvolve, resample_poly

FS = 8000
C = 343.0
SHIFT = 16          # samples of pre-roll kept for the arrival at the microphone nearest the source
TAPS = 64           # length of one scattered arrival


class Housing:
    """BEM surface pressure of the four microphone panels for 65,160 plane-wave directions, 0 to 4 kHz."""

    def __init__(self, table: Path) -> None:
        with h5py.File(table, "r") as f:
            self.pressure = np.asarray(f["p"]).transpose(1, 2, 0)        # [direction, frequency, mic]
            self.grid = np.asarray(f["doa_grid"], dtype=np.float64)       # unit vectors toward the source
            self.freqs = np.asarray(f["freqs"], dtype=np.float64)
            self.mics = np.asarray(f["centroids"], dtype=np.float64)      # panel centers, m
        self.n_az, self.n_el = 360, 181

    def nearest(self, unit: np.ndarray) -> np.ndarray:
        az = np.degrees(np.arctan2(unit[..., 1], unit[..., 0])) % 360.0
        el = np.degrees(np.arcsin(np.clip(unit[..., 2], -1, 1)))
        ia = np.rint(az).astype(int) % self.n_az
        ie = np.clip(np.rint(el + 90.0).astype(int), 0, self.n_el - 1)
        return ia * self.n_el + ie

    def arrival(self, index: int, fractional: float) -> np.ndarray:
        """Impulse response of one arrival, [TAPS, mic], delayed by SHIFT + fractional samples."""
        phase = np.exp(-2j * np.pi * self.freqs[:, None] * (SHIFT + fractional) / FS)
        response = np.fft.irfft(self.pressure[index] * phase, n=512, axis=0)
        return response[:TAPS]


@dataclass
class Room:
    size: np.ndarray        # m
    t60: float              # s
    device: np.ndarray      # array center, m

    def reflection(self) -> float:
        lx, ly, lz = self.size
        volume, surface = lx * ly * lz, 2 * (lx * ly + ly * lz + lx * lz)
        alpha = min(0.99, 0.161 * volume / (surface * self.t60))
        return float(np.sqrt(1.0 - alpha))


def impulse_response(housing: Housing, room: Room, source: np.ndarray, max_order: int = 10,
                     length_s: float | None = None) -> np.ndarray:
    """Room impulse response at the four panels, [samples, mic]."""
    beta = room.reflection()
    length = int(FS * (length_s or min(1.2, room.t60 * 1.2))) + TAPS
    response = np.zeros((length, 4))
    axes = []
    for axis in range(3):
        L, s = room.size[axis], source[axis]
        entries = []
        for n in range(-max_order, max_order + 1):
            for q in (0, 1):
                order = abs(2 * n - q)
                if order <= max_order:
                    entries.append(((1 - 2 * q) * s + 2 * n * L, order))
        axes.append(entries)
    positions, orders = [], []
    for x, ox in axes[0]:
        for y, oy in axes[1]:
            for z, oz in axes[2]:
                if ox + oy + oz <= max_order:
                    positions.append((x, y, z))
                    orders.append(ox + oy + oz)
    positions = np.asarray(positions) - room.device
    orders = np.asarray(orders)
    distance = np.linalg.norm(positions, axis=1)
    delay = distance / C * FS
    keep = delay + TAPS < length - SHIFT
    positions, orders, distance, delay = positions[keep], orders[keep], distance[keep], delay[keep]
    gain = beta ** orders / distance
    indices = housing.nearest(positions / distance[:, None])
    for idx, g, d in zip(indices, gain, delay):
        whole = int(np.floor(d))
        start = whole - SHIFT
        part = housing.arrival(idx, d - whole) * g
        if start < 0:
            part, start = part[-start:], 0
        response[start:start + part.shape[0]] += part
    return response


def diffuse_noise(housing: Housing, samples: int, rng: np.random.Generator, n_sources: int = 64) -> np.ndarray:
    """Pink noise from directions spread over the sphere, each through the housing."""
    noise = np.zeros((samples, 4))
    units = rng.normal(size=(n_sources, 3))
    units /= np.linalg.norm(units, axis=1, keepdims=True)
    freqs = np.fft.rfftfreq(samples, 1 / FS)
    shaping = 1 / np.sqrt(np.maximum(freqs, 50.0))
    for unit in units:
        white = np.fft.rfft(rng.normal(size=samples))
        pink = np.fft.irfft(white * shaping, n=samples)
        h = housing.arrival(int(housing.nearest(unit)), 0.0)
        noise += fftconvolve(pink[:, None], h, axes=0)[SHIFT:SHIFT + samples]
    return noise / np.sqrt(n_sources)


def load_speech(path: Path) -> np.ndarray:
    import soundfile as sf
    audio, rate = sf.read(str(path), dtype="float64", always_2d=True)
    audio = audio[:, 0]
    if rate != FS:
        audio = resample_poly(audio, FS, rate)
    return audio / (np.sqrt(np.mean(audio ** 2)) + 1e-12)


def unit_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def random_room(rng: np.random.Generator, t60: float) -> Room:
    size = np.array([rng.uniform(5, 9), rng.uniform(5, 9), rng.uniform(2.6, 3.2)])
    device = np.array([size[0] / 2 + rng.uniform(-0.5, 0.5), rng.uniform(1.2, 1.8), 0.85])
    return Room(size=size, t60=t60, device=device)


def place_talker(rng: np.random.Generator, room: Room, azimuth_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """A talker at the given azimuth, 1.5 to 3 m away, mouth 1.1 to 1.7 m high, inside the room."""
    for _ in range(200):
        distance = rng.uniform(1.5, 3.0)
        mouth = rng.uniform(1.1, 1.7)
        horizontal = np.sqrt(max(distance ** 2 - (mouth - room.device[2]) ** 2, 0.5))
        a = np.radians(azimuth_deg)
        position = room.device + np.array([horizontal * np.cos(a), horizontal * np.sin(a), 0.0])
        position[2] = mouth
        if np.all(position[:2] > 0.5) and np.all(position[:2] < room.size[:2] - 0.5):
            direction = position - room.device
            return position, direction / np.linalg.norm(direction)
    raise RuntimeError("no talker position fits in the room")
