"""Diffuse noise with the coherence that the housing gives it, and sensor noise."""
from __future__ import annotations

import math

import numpy as np
import scipy.linalg
import scipy.signal

C_SOUND = 343.0


def body_coherence(bem_table, panel_indices: np.ndarray, freq_indices: np.ndarray,
                   ) -> np.ndarray:
    """(M, M, len(freq_indices)) diffuse-field coherence of the body."""
    response = np.asarray(bem_table.get_panels(np.asarray(panel_indices)))
    response = response[:, :, freq_indices].astype(np.complex128)

    n_az, n_el = int(bem_table.n_az), int(bem_table.n_el)
    elevation = np.deg2rad(np.linspace(-90.0, 90.0, n_el))
    weight = np.repeat(np.cos(elevation)[None, :], n_az, axis=0).reshape(-1)
    weight = weight / weight.sum()

    weighted = response * weight[None, :, None]
    gram = np.einsum("mdf,ndf->mnf", weighted, response.conj())
    diagonal = np.sqrt(np.real(np.einsum("mmf->mf", gram)))
    gamma = gram / (diagonal[:, None, :] * diagonal[None, :, :] + 1e-30)
    return gamma


def free_field_coherence(positions: np.ndarray, frequencies: np.ndarray,
                         c: float = C_SOUND) -> np.ndarray:
    """(M, M, F) spherically isotropic free-field coherence, sinc(kd)."""
    distance = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    argument = 2.0 * np.pi * frequencies[None, None, :] * distance[:, :, None] / c
    return np.sinc(argument / np.pi)


class DiffuseMixer:
    """Applies a fixed coherence to independent noise channels via Cholesky."""

    def __init__(self, coherence: np.ndarray, n_fft: int) -> None:
        n_mic = coherence.shape[0]
        self.n_fft = n_fft
        self.factors = np.zeros_like(coherence)
        for k in range(coherence.shape[2]):
            matrix = coherence[:, :, k] + 1e-9 * np.eye(n_mic)
            try:
                self.factors[:, :, k] = scipy.linalg.cholesky(matrix)
            except Exception:
                values, vectors = np.linalg.eigh(matrix)
                values = np.clip(values.real, 0.0, None)
                self.factors[:, :, k] = (vectors * np.sqrt(values)).conj().T

    def __call__(self, mono: np.ndarray, n_samples: int, n_mic: int,
                 rng=None) -> np.ndarray:
        """(n_samples, n_mic) diffuse noise carrying `mono`'s spectrum."""
        n_fft = self.n_fft
        needed = n_samples * n_mic
        if len(mono) < needed:
            mono = np.tile(mono, int(np.ceil(needed / max(len(mono), 1))) + 1)
        span = len(mono) - needed
        start = int(rng.integers(0, span + 1)) if (rng is not None and span > 0) else 0
        channels = np.zeros((n_samples, n_mic))
        for m in range(n_mic):
            begin = start + m * n_samples
            segment = mono[begin:begin + n_samples]
            channels[:, m] = segment - segment.mean()

        padded = np.vstack([np.zeros((n_fft // 2, n_mic)), channels,
                            np.zeros((n_fft // 2, n_mic))])
        _, _, spectra = scipy.signal.stft(padded.T, window="hann", nperseg=n_fft,
                                          noverlap=int(0.75 * n_fft), nfft=n_fft)
        mixed = np.zeros(spectra.shape, dtype=complex)
        for k in range(1, n_fft // 2 + 1):
            mixed[:, k, :] = (spectra[:, k, :].T @ self.factors[:, :, k].conj()).T
        _, waveform = scipy.signal.istft(mixed, window="hann", nperseg=n_fft,
                                         noverlap=int(0.75 * n_fft), nfft=n_fft)
        waveform = waveform.T[n_fft // 2: -n_fft // 2, :]
        return np.real(waveform[:n_samples, :])


def build_mixer(bem_table, panel_indices: np.ndarray, fs: int, n_fft: int,
                use_body: bool = True) -> DiffuseMixer:
    """Coherence at the mixer's own STFT resolution, taken from the BEM table."""
    table_freqs = np.asarray(bem_table.freqs, dtype=np.float64)
    target = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    freq_indices = np.abs(table_freqs[None, :] - target[:, None]).argmin(axis=1)
    if use_body:
        coherence = body_coherence(bem_table, panel_indices, freq_indices)
    else:
        positions = np.asarray(bem_table.centroids)[np.asarray(panel_indices)]
        coherence = free_field_coherence(positions, target)
    return DiffuseMixer(coherence, n_fft)


def report(bem_table, panel_indices, fs: int, n_fft: int) -> None:
    """Sanity check: body coherence must approach sinc(kd) as frequency drops."""
    table_freqs = np.asarray(bem_table.freqs, dtype=np.float64)
    target = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    freq_indices = np.abs(table_freqs[None, :] - target[:, None]).argmin(axis=1)
    body = body_coherence(bem_table, panel_indices, freq_indices)
    positions = np.asarray(bem_table.centroids)[np.asarray(panel_indices)]
    free = free_field_coherence(positions, target)
    upper = np.triu_indices(body.shape[0], 1)
    print(f"{'f[Hz]':>7} {'max body-free':>16} {'min Re body':>12} {'min sinc':>10}")
    for hz in (50, 125, 250, 500, 1000, 2000, 3900):
        k = int(np.abs(target - hz).argmin())
        print(f"{target[k]:7.0f} {np.abs(body[:, :, k] - free[:, :, k])[upper].max():16.3f} "
              f"{np.real(body[:, :, k])[upper].min():12.3f} "
              f"{np.real(free[:, :, k])[upper].min():10.3f}")
