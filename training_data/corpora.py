"""Speech, noise and sound-event corpora, split by speaker."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

LIBRI_ROOT = Path(os.environ.get("LIBRISPEECH", "LibriSpeech"))
ZEROTH_ROOT = Path(os.environ.get("ZEROTH_KOREAN", "zeroth_korean"))
NOISE_DIR = Path(os.environ.get("NOISEX92", "noise92"))
ESC_ROOT = Path(os.environ.get("ESC50", "ESC-50-master"))

LIBRI_SUBSET = {"train": "train-clean-100", "dev": "dev-clean", "test": "test-clean"}
ESC_FOLDS = {"train": {1, 2, 3}, "dev": {4}, "test": {5}}
ESC_CLASSES = ["coughing", "sneezing", "door_wood_knock", "door_wood_creaks",
               "keyboard_typing", "mouse_click", "footsteps", "clapping"]
ZEROTH_DEV_SPEAKERS = 10


def _by_speaker(paths, speaker_of) -> dict[str, list[str]]:
    table: dict[str, list[str]] = {}
    for path in paths:
        table.setdefault(speaker_of(path), []).append(str(path))
    return {k: sorted(v) for k, v in sorted(table.items())}


def librispeech(split: str) -> dict[str, list[str]]:
    root = LIBRI_ROOT / LIBRI_SUBSET[split]
    return _by_speaker(root.rglob("*.flac"), lambda p: f"libri-{p.parts[-3]}")


def zeroth(split: str) -> dict[str, list[str]]:
    """train_data_01 is split by speaker; test_data_01 is the official test set."""
    if split == "test":
        root = ZEROTH_ROOT / "test_data_01"
        return _by_speaker(root.rglob("*.flac"), lambda p: f"zeroth-{p.parts[-2]}")
    table = _by_speaker((ZEROTH_ROOT / "train_data_01").rglob("*.flac"),
                        lambda p: f"zeroth-{p.parts[-2]}")
    speakers = sorted(table)
    held_out = set(speakers[-ZEROTH_DEV_SPEAKERS:])
    keep = held_out if split == "dev" else set(speakers) - held_out
    return {k: v for k, v in table.items() if k in keep}


class SpeechPool:
    """Two corpora, drawn 50/50, with distinct speakers inside one clip."""

    def __init__(self, split: str) -> None:
        self.corpora = [librispeech(split), zeroth(split)]
        self.speakers = [sorted(c) for c in self.corpora]
        if not all(self.speakers):
            raise SystemExit(f"{split}: a corpus has no speakers")

    def describe(self) -> str:
        return "  ".join(
            f"{name} {len(s)} speakers {sum(len(self.corpora[i][k]) for k in s)} utterances"
            for i, (name, s) in enumerate(zip(("LibriSpeech", "Zeroth"), self.speakers)))

    def pick_speakers(self, rng, count: int) -> list[tuple[int, str]]:
        chosen: list[tuple[int, str]] = []
        while len(chosen) < count:
            corpus = int(rng.integers(len(self.corpora)))
            speaker = self.speakers[corpus][int(rng.integers(len(self.speakers[corpus])))]
            if (corpus, speaker) not in chosen:
                chosen.append((corpus, speaker))
        return chosen

    def segment(self, rng, who: tuple[int, str], n_samples: int, fs: int) -> np.ndarray:
        corpus, speaker = who
        files = self.corpora[corpus][speaker]
        for _ in range(50):
            path = files[int(rng.integers(len(files)))]
            wave = load_mono(path, fs)
            if len(wave) < n_samples + 1:
                wave = np.tile(wave, int(np.ceil((n_samples + 1) / max(len(wave), 1))))
            start = int(rng.integers(0, len(wave) - n_samples))
            segment = wave[start:start + n_samples]
            if float(np.std(segment)) >= 1e-4:
                return segment.astype(np.float64)
        return np.zeros(n_samples, dtype=np.float64)


def load_mono(path: str | Path, fs: int) -> np.ndarray:
    wave, rate = sf.read(str(path), dtype="float64")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if rate != fs:
        divisor = np.gcd(int(rate), int(fs))
        wave = scipy.signal.resample_poly(wave, fs // divisor, int(rate) // divisor)
    return wave


def noise_files() -> list[str]:
    return sorted(str(p) for p in NOISE_DIR.iterdir()
                  if p.suffix in (".wav", ".flac", ".ogg"))


def esc_index(split: str) -> dict[str, list[str]]:
    """ESC-50 folds keep recordings of one original source in one split."""
    meta = ESC_ROOT / "meta" / "esc50.csv"
    if not meta.exists():
        return {}
    folds = ESC_FOLDS[split]
    index: dict[str, list[str]] = {c: [] for c in ESC_CLASSES}
    with open(meta) as handle:
        for row in csv.DictReader(handle):
            if row["category"] in index and int(row["fold"]) in folds:
                index[row["category"]].append(str(ESC_ROOT / "audio" / row["filename"]))
    return {k: v for k, v in index.items() if v}


def esc_clip(rng, index: dict[str, list[str]], n_samples: int, fs: int) -> np.ndarray:
    keys = list(index)
    files = index[keys[int(rng.integers(len(keys)))]]
    wave = load_mono(files[int(rng.integers(len(files)))], fs)
    window = max(int(0.05 * fs), 1)
    smooth = np.convolve(np.abs(wave), np.ones(window) / window, mode="same")
    active = np.flatnonzero(smooth > 0.1 * (smooth.max() + 1e-12))
    if active.size:
        wave = wave[active[0]:active[-1] + 1]
    if len(wave) < n_samples:
        wave = np.tile(wave, int(np.ceil(n_samples / max(len(wave), 1))))
    return wave[:n_samples]
