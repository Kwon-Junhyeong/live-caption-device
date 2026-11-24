from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import stft
from torch.utils.data import Dataset


def stft_multichannel(audio: np.ndarray, sample_rate: int, n_fft: int, hop: int) -> np.ndarray:
    _, _, spectra = stft(np.asarray(audio).T, fs=sample_rate, window="hann", nperseg=n_fft,
                         noverlap=n_fft - hop, nfft=n_fft, boundary="zeros", padded=True, axis=-1)
    return np.asarray(spectra)


@dataclass(frozen=True)
class LoaderConfig:
    sample_rate: int = 8000
    n_fft: int = 256
    hop: int = 128
    crop_seconds: float = 4.0
    max_sources: int = 3
    activity_threshold: float = 0.05
    target_key: str = "src_direct"
    band_lo_hz: float = 150.0
    band_hi_hz: float = 3800.0
    # 0: one level per clip. > 0: the causal running level of the device (streaming_net._normalise)
    level_halflife_sec: float = 0.0
    level_block_frames: int = 8

    def band_mask(self) -> np.ndarray:
        frequencies = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)
        return (frequencies >= self.band_lo_hz) & (frequencies <= self.band_hi_hz)


class MultiSourceDataset(Dataset):
    def __init__(self, root: str | Path, config: LoaderConfig | None = None,
                 train: bool = True) -> None:
        self.root = Path(root)
        self.config = config or LoaderConfig()
        self.train = train
        self.paths = sorted(self.root.glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"no clips in {self.root}")
        self.mask = self.config.band_mask()[None, :, None]

    def __len__(self) -> int:
        return len(self.paths)

    def _level(self, spectra: np.ndarray) -> np.ndarray:
        config = self.config
        magnitude = np.abs(spectra[0])
        n_frame = magnitude.shape[1]
        if config.level_halflife_sec <= 0.0:
            return np.full(n_frame, max(float(magnitude.mean()), 1e-12), dtype=np.float32)

        block = max(1, int(config.level_block_frames))
        seconds = block * config.hop / config.sample_rate
        decay = 0.5 ** (seconds / config.level_halflife_sec)
        level = np.empty(n_frame, dtype=np.float32)
        running = None
        for start in range(0, n_frame, block):
            stop = min(start + block, n_frame)
            current = float(magnitude[:, start:stop].mean())
            running = current if running is None else decay * running + (1.0 - decay) * current
            level[start:stop] = max(running, 1e-12)
        return level

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        config = self.config
        path = self.paths[index]
        meta = np.load(path, allow_pickle=True)
        pcm, _ = sf.read(str(path.with_suffix(".wav")), dtype="int16")

        divisor = 32767.0 * float(meta["wav_scale"])
        mixture = pcm.astype(np.float32) / divisor
        direct = meta["src_direct"].astype(np.float32) / divisor
        wanted = (direct if config.target_key == "src_direct"
                  else meta[config.target_key].astype(np.float32) / divisor)

        crop = int(round(config.crop_seconds * config.sample_rate))
        if mixture.shape[0] < crop:
            raise ValueError(f"{path.name}: clip shorter than the crop")
        start = int(np.random.randint(0, mixture.shape[0] - crop + 1)) if self.train else 0
        mixture = mixture[start:start + crop]
        direct = direct[:, start:start + crop]
        wanted = wanted[:, start:start + crop]

        spectra = stft_multichannel(mixture, config.sample_rate, config.n_fft, config.hop)

        def source_stft(rows: np.ndarray) -> np.ndarray:
            if rows.shape[0] == 0:                     # clips with no speaker
                return np.zeros((0,) + spectra.shape[1:], dtype=spectra.dtype)
            return stft_multichannel(rows.T, config.sample_rate, config.n_fft, config.hop)

        direct_spectra = source_stft(direct)
        target_spectra = (direct_spectra if config.target_key == "src_direct"
                          else source_stft(wanted))

        spectra = spectra * self.mask
        target_spectra = target_spectra * self.mask
        direct_spectra = direct_spectra * self.mask

        level = self._level(spectra)
        spectra = spectra / level[None, None, :]
        target_spectra = target_spectra / level[None, None, :]

        reference = np.abs(spectra[0]) * level[None, :]
        activity = (np.abs(direct_spectra) / (reference[None] + 1e-12)).mean(axis=1)
        activity = (activity > config.activity_threshold).astype(np.float32)

        n_source, n_freq, n_frame = target_spectra.shape
        slots = config.max_sources
        if n_source > slots:
            raise ValueError(f"{path.name}: {n_source} sources > {slots} slots")

        directions = np.zeros((slots, 3), dtype=np.float32)
        directions[:n_source] = np.atleast_2d(meta["d_unit"]).astype(np.float32)
        padded_activity = np.zeros((slots, n_frame), dtype=np.float32)
        padded_activity[:n_source] = activity
        padded_target = np.zeros((slots, n_freq, n_frame), dtype=np.complex64)
        padded_target[:n_source] = target_spectra

        return {
            "mixture": torch.from_numpy(np.stack([spectra.real, spectra.imag]).astype(np.float32)),
            "target": torch.from_numpy(np.stack([padded_target.real,
                                                 padded_target.imag]).astype(np.float32)),
            "direction": torch.from_numpy(directions),
            "activity": torch.from_numpy(padded_activity),
            "n_src": torch.tensor(int(meta["n_src"]), dtype=torch.long),
            "level": torch.tensor(level, dtype=torch.float32),
        }


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    mixture = torch.stack([item["mixture"] for item in batch])
    return {
        "mixture": mixture.flatten(1, 2),
        "target": torch.stack([item["target"] for item in batch]).permute(0, 2, 1, 3, 4),
        "direction": torch.stack([item["direction"] for item in batch]),
        "activity": torch.stack([item["activity"] for item in batch]),
        "n_src": torch.stack([item["n_src"] for item in batch]),
        "level": torch.stack([item["level"] for item in batch]),
    }
