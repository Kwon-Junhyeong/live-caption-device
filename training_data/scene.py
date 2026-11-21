"""Geometry of one scene: the room, the device on a desk, seated and standing speakers and their separation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyroomacoustics as pra

ROOM_MIN = np.array([5.0, 5.0, 2.4])
ROOM_MAX = np.array([12.0, 12.0, 3.5])
T60_MIN, T60_MAX = 0.2, 1.3
ALPHA_MIN, ALPHA_MAX = 0.08, 0.35

MIC_HEIGHT_MIN, MIC_HEIGHT_MAX = 0.45, 1.25

DIST_MIN, DIST_MAX = 1.5, 4.5

AZ_MIN, AZ_MAX = 30.0, 150.0
EL_MIN, EL_MAX = -10.0, 35.0
AZ_LIMIT = 90.0
MIN_SEP_DEG = 7.0
MIN_SEP_AZ_DEG = 7.0
SEP_MEDIAN_DEG = 15.0
SEP_SIGMA = 0.558

ARRAY_CLEARANCE = 0.8
SOURCE_CLEARANCE = 0.5


@dataclass
class Source:
    az_deg: float
    el_deg: float
    horizontal_m: float
    distance_m: float
    mouth_m: float
    standing: bool
    direction: np.ndarray
    position: np.ndarray


@dataclass
class Scene:
    room_sz: np.ndarray
    t60: float
    snr_db: float
    mic_height_m: float
    body_center: np.ndarray
    sources: list[Source] = field(default_factory=list)
    attempts: int = 1


def unit(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def separation_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


def _room(rng) -> tuple[np.ndarray, float]:
    for _ in range(50):
        size = rng.uniform(ROOM_MIN, ROOM_MAX)
        volume = float(np.prod(size))
        surface = 2.0 * (size[0] * size[1] + size[1] * size[2] + size[2] * size[0])
        alpha = float(rng.uniform(ALPHA_MIN, ALPHA_MAX))
        t60 = 0.161 * volume / (surface * alpha)
        if not (T60_MIN <= t60 <= T60_MAX):
            continue
        try:
            pra.inverse_sabine(t60, list(size))
            return size, float(t60)
        except Exception:
            continue
    return ROOM_MIN.copy(), 0.6


def _max_horizontal(center: np.ndarray, az_deg: float, room: np.ndarray) -> float:
    """Furthest the source may sit along this azimuth before leaving the room."""
    limit = DIST_MAX
    for axis, component in enumerate((np.cos(np.radians(az_deg)),
                                      np.sin(np.radians(az_deg)))):
        if abs(component) < 1e-9:
            continue
        bound = (room[axis] - SOURCE_CLEARANCE) if component > 0 else SOURCE_CLEARANCE
        limit = min(limit, (bound - center[axis]) / component)
    return float(limit)


def _place(rng, center, az_deg, room, mic_height, el_deg=None) -> Source | None:
    reach = _max_horizontal(center, az_deg, room)
    if reach < DIST_MIN:
        return None
    if el_deg is None:
        el_deg = float(rng.uniform(EL_MIN, EL_MAX))

    slope = np.tan(np.radians(el_deg))
    ceiling = room[2] - SOURCE_CLEARANCE
    floor = SOURCE_CLEARANCE
    if slope > 1e-6:
        vertical_reach = (ceiling - mic_height) / slope
    elif slope < -1e-6:
        vertical_reach = (mic_height - floor) / (-slope)
    else:
        vertical_reach = DIST_MAX
    upper = min(DIST_MAX, reach, vertical_reach)
    if upper < DIST_MIN:
        return None

    horizontal = float(rng.uniform(DIST_MIN, upper))
    mouth = mic_height + horizontal * slope
    standing = bool(mouth >= 1.30)
    position = np.array([center[0] + horizontal * np.cos(np.radians(az_deg)),
                         center[1] + horizontal * np.sin(np.radians(az_deg)),
                         mouth])
    offset = position - center
    distance = float(np.linalg.norm(offset))
    return Source(az_deg=az_deg, el_deg=el_deg, horizontal_m=horizontal,
                  distance_m=distance, mouth_m=mouth, standing=standing,
                  direction=offset / distance, position=position)


def sample_scene(rng, n_sources: int, snr_min: float, snr_max: float,
                 az_limit: float = AZ_LIMIT, min_sep_deg: float = MIN_SEP_DEG,
                 az_min: float = AZ_MIN, az_max: float = AZ_MAX,
                 min_sep_az_deg: float = MIN_SEP_AZ_DEG,
                 sep_median_deg: float = SEP_MEDIAN_DEG,
                 sep_sigma: float = SEP_SIGMA) -> Scene:
    mu = np.log(sep_median_deg)
    for attempt in range(1, 61):
        room, t60 = _room(rng)
        mic_height = float(rng.uniform(MIC_HEIGHT_MIN, MIC_HEIGHT_MAX))
        center = np.array([
            float(rng.uniform(ARRAY_CLEARANCE, room[0] - ARRAY_CLEARANCE)),
            float(rng.uniform(ARRAY_CLEARANCE, room[1] - ARRAY_CLEARANCE)),
            mic_height])
        scene = Scene(room_sz=room, t60=t60,
                      snr_db=float(rng.uniform(snr_min, snr_max)),
                      mic_height_m=mic_height, body_center=center,
                      attempts=attempt)
        if n_sources == 0:
            return scene

        first = _place(rng, center, float(rng.uniform(az_min, az_max)), room,
                       mic_height)
        if first is None:
            continue
        scene.sources.append(first)
        while len(scene.sources) < n_sources:
            for _ in range(400):
                gap = float(np.clip(rng.lognormal(mu, sep_sigma), min_sep_deg, 120.0))
                anchor = scene.sources[int(rng.integers(len(scene.sources)))]
                az = anchor.az_deg + gap * float(rng.choice((-1.0, 1.0)))
                if not (az_min <= az <= az_max):
                    continue
                candidate = _place(rng, center, az, room, mic_height)
                if candidate is None:
                    continue
                if any(separation_deg(candidate.direction, other.direction) < min_sep_deg
                       for other in scene.sources):
                    continue
                if any(abs(candidate.az_deg - other.az_deg) < min_sep_az_deg
                       for other in scene.sources):
                    continue
                scene.sources.append(candidate)
                break
            else:
                break
        if len(scene.sources) == n_sources:
            return scene
    raise RuntimeError(f"could not place {n_sources} sources")


def sample_off_target(rng, scene: Scene, rear: bool, az_limit: float = AZ_LIMIT,
                      min_sep_deg: float = MIN_SEP_DEG,
                      az_min: float = AZ_MIN, az_max: float = AZ_MAX) -> Source | None:
    """A speaker outside the camera view, or a non-speech source inside it."""
    for _ in range(400):
        az = (float(rng.uniform(az_max, az_min + 360.0)) if rear
              else float(rng.uniform(az_min, az_max)))
        az = az - 360.0 if az > 180.0 else az
        candidate = _place(rng, scene.body_center, az, scene.room_sz,
                           scene.body_center[2])
        if candidate is None:
            continue
        if any(separation_deg(candidate.direction, other.direction) < min_sep_deg
               for other in scene.sources):
            continue
        return candidate
    return None
