"""Training scenes for the device at 8 kHz: up to three speakers in a room, heard through the housing, with noise."""
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bem_table import BEMTableLazy                                    # noqa: E402
from noise import acoustic_power                                      # noqa: E402
from synthesize_reverb import synthesize_reverb                       # noqa: E402

import body_noise                                                     # noqa: E402
import corpora                                                        # noqa: E402
import scene as scene_mod                                             # noqa: E402

FS = 8000
N_FFT_NET, HOP_NET = 256, 128
N_FFT_BEM = 512
PRA_MAX_ORDER = 17
RENDER_APPLY_CONJUGATE = False

BODY_ID = "devicefin4k"
PANEL_INDICES = np.array([0, 1, 2, 3], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="multi-speaker training scenes for the device, 8 kHz")
    p.add_argument("bem", type=Path, help="BEM table of the housing (HDF5)")
    p.add_argument("out_dir", type=Path)
    p.add_argument("-n", "--n_samples", type=int, default=100)
    p.add_argument("--split", default="train", choices=["train", "dev", "test"])
    p.add_argument("--seed", type=int, default=20260827)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--tag", default=None, help="name of the output subfolder (default: the split)")

    p.add_argument("--k_max", type=int, default=3)
    p.add_argument("--p_silence", type=float, default=0.12)
    p.add_argument("--t_sec", type=float, default=4.0)
    p.add_argument("--level_db", type=float, default=5.0)
    p.add_argument("--gate_prob", type=float, default=0.40)
    p.add_argument("--gate_span_sec", type=float, default=6.0,
                   help="reference length of one utterance span (s); smaller gives more pauses in a clip")

    p.add_argument("--snr_min", type=float, default=0.0)
    p.add_argument("--snr_max", type=float, default=25.0)
    p.add_argument("--channel_gain_db", type=float, default=4.0)
    p.add_argument("--sensor_ratio_min", type=float, default=10.0)
    p.add_argument("--sensor_ratio_max", type=float, default=32.0)
    p.add_argument("--noise_tilt_min", type=float, default=-8.0)
    p.add_argument("--noise_tilt_max", type=float, default=0.0)

    p.add_argument("--rear_prob", type=float, default=0.20)
    p.add_argument("--interferer_prob", type=float, default=0.25)
    p.add_argument("--interferer_sir_min", type=float, default=-10.0)
    p.add_argument("--interferer_sir_max", type=float, default=20.0)

    p.add_argument("--min_sep_deg", type=float, default=scene_mod.MIN_SEP_DEG)
    p.add_argument("--min_sep_az_deg", type=float, default=scene_mod.MIN_SEP_AZ_DEG,
                   help="least azimuth separation; a 3D limit alone lets pairs pass on elevation")
    p.add_argument("--sep_median_deg", type=float, default=scene_mod.SEP_MEDIAN_DEG)
    p.add_argument("--sep_sigma", type=float, default=scene_mod.SEP_SIGMA)
    p.add_argument("--az_limit_deg", type=float, default=scene_mod.AZ_LIMIT)

    p.add_argument("--store_ref", action="store_true",
                   help="also store the reverberant targets (src_ref)")
    p.add_argument("--free_field_noise", action="store_true",
                   help="diffuse noise without the housing's scattering (for comparison)")
    p.add_argument("--selfcheck", type=int, default=0,
                   help="draw the geometry only, without the BEM table or audio")
    return p.parse_args()


class CachedPanels:
    """BEMTableLazy.get_panels copies the whole array on every call; render() hits
    it several times per clip, so hold the one panel set we ever ask for."""

    def __init__(self, table, panel_indices: np.ndarray) -> None:
        self._table = table
        self._key = tuple(int(i) for i in np.atleast_1d(panel_indices))
        self._cache = np.ascontiguousarray(table.get_panels(np.asarray(panel_indices)))

    def get_panels(self, panel_indices):
        if tuple(int(i) for i in np.atleast_1d(panel_indices)) == self._key:
            return self._cache
        return self._table.get_panels(panel_indices)

    def __getattr__(self, name):
        return getattr(self._table, name)


def gate_segment(rng, signal: np.ndarray, t_samples: int, seconds: float,
                 span_seconds: float = 6.0) -> np.ndarray:
    spans = max(1, int(round(seconds / span_seconds)))
    window = t_samples // spans
    ramp = max(int(0.02 * window), 1)
    gated = np.zeros_like(signal)
    for index in range(spans):
        base = index * window
        length = int(rng.uniform(0.35, 0.85) * window)
        if length <= 2 * ramp:
            continue
        start = base + int(rng.integers(0, max(window - length, 1)))
        shape = np.ones(length)
        shape[:ramp] = np.linspace(0.0, 1.0, ramp)
        shape[-ramp:] = np.linspace(1.0, 0.0, ramp)
        gated[start:start + length] = signal[start:start + length] * shape
    return gated


def _apply_tilt(mono: np.ndarray, fs: int, db_per_octave: float) -> np.ndarray:
    if abs(db_per_octave) < 1e-6:
        return mono
    n = len(mono)
    spectrum = np.fft.rfft(mono)
    freq = np.fft.rfftfreq(n, 1.0 / fs)
    octaves = np.log2(np.maximum(freq, 20.0) / 1000.0)
    spectrum *= 10 ** (db_per_octave * octaves / 20.0)
    out = np.fft.irfft(spectrum, n=n)
    scale = np.sqrt((mono ** 2).mean() / max((out ** 2).mean(), 1e-20))
    return out * scale


def mean_power(signal: np.ndarray) -> float:
    return float(np.mean([acoustic_power(signal[:, c]) for c in range(signal.shape[1])]))


def draw_count(rng, k_max: int, p_silence: float) -> int:
    if rng.random() < p_silence:
        return 0
    return int(rng.integers(1, k_max + 1))


def synthesize(seed, bem, speech, esc_files, noise_paths, mixer, panel_pos,
               t_samples, args) -> dict:
    rng = np.random.default_rng(seed)
    n_sources = draw_count(rng, args.k_max, args.p_silence)
    layout = scene_mod.sample_scene(
        rng, n_sources, args.snr_min, args.snr_max, az_limit=args.az_limit_deg,
        min_sep_deg=args.min_sep_deg,
        min_sep_az_deg=args.min_sep_az_deg, sep_median_deg=args.sep_median_deg,
        sep_sigma=args.sep_sigma)

    def render(mono, position, order):
        return synthesize_reverb(
            source=mono, source_pos=position, body_center=layout.body_center,
            panel_indices=PANEL_INDICES, room_dim=tuple(layout.room_sz),
            rt60=layout.t60, bem_table=bem, fs=FS, n_fft=N_FFT_BEM, hop=HOP_NET,
            max_order=order)

    who = speech.pick_speakers(rng, n_sources)
    images, directs, gains_db, gated_flags = [], [], [], []
    reference_power = 1.0
    for index, source in enumerate(layout.sources):
        mono = speech.segment(rng, who[index], t_samples, FS)
        gated = bool(rng.random() < args.gate_prob)
        if gated:
            mono = gate_segment(rng, mono, t_samples, args.t_sec, args.gate_span_sec)
        gated_flags.append(gated)
        image = render(mono, source.position, PRA_MAX_ORDER)
        direct = render(mono, source.position, 0)
        if float(np.std(image)) < 1e-6 or float(np.std(direct)) < 1e-6:
            raise RuntimeError(f"render failed, seed={seed} src={index}")
        if index == 0:
            reference_power = mean_power(image)
            gain, target_db = 1.0, 0.0
        else:
            target_db = float(rng.uniform(-args.level_db, args.level_db))
            gain = float(np.sqrt(reference_power / mean_power(image)
                                 * 10 ** (target_db / 10)))
        gains_db.append(target_db)
        images.append(image * gain)
        directs.append(direct * gain)

    n_mic = len(PANEL_INDICES)
    mixture_clean = (np.sum(images, axis=0) if images
                     else np.zeros((t_samples, n_mic)))
    direct_clean = (np.sum(directs, axis=0) if directs
                    else np.zeros((t_samples, n_mic)))

    noise_mono = corpora.load_mono(noise_paths[int(rng.integers(len(noise_paths)))], FS)
    tilt_db_oct = float(rng.uniform(args.noise_tilt_min, args.noise_tilt_max))
    noise_mono = _apply_tilt(noise_mono, FS, tilt_db_oct)
    diffuse = mixer(noise_mono, t_samples, n_mic, rng=rng)
    diffuse_power = float(np.mean(diffuse ** 2)) + 1e-20
    sensor_ratio_db = float(rng.uniform(args.sensor_ratio_min, args.sensor_ratio_max))
    sensor = rng.standard_normal((t_samples, n_mic))
    sensor *= np.sqrt(diffuse_power / 10 ** (sensor_ratio_db / 10)
                      / (float(np.mean(sensor ** 2)) + 1e-20))
    noise = diffuse + sensor

    interferers = []
    if rng.random() < args.rear_prob:
        placed = scene_mod.sample_off_target(rng, layout, rear=True,
                                             az_limit=args.az_limit_deg,
                                             min_sep_deg=args.min_sep_deg)
        if placed is not None:
            extra = speech.pick_speakers(rng, 1)[0]
            interferers.append(("rear_speech",
                                speech.segment(rng, extra, t_samples, FS), placed))
    if rng.random() < args.interferer_prob and esc_files:
        placed = scene_mod.sample_off_target(rng, layout, rear=False,
                                             az_limit=args.az_limit_deg,
                                             min_sep_deg=args.min_sep_deg)
        if placed is not None:
            interferers.append(("nonspeech",
                                corpora.esc_clip(rng, esc_files, t_samples, FS), placed))

    interferer_meta = []
    for kind, mono, placed in interferers:
        rendered = render(mono, placed.position, PRA_MAX_ORDER)
        if float(np.std(rendered)) < 1e-6:
            continue
        sir_db = float(rng.uniform(args.interferer_sir_min, args.interferer_sir_max))
        anchor = reference_power if n_sources else mean_power(noise)
        gain = float(np.sqrt(anchor / (mean_power(rendered) + 1e-12)
                             * 10 ** (-sir_db / 10)))
        mixture_clean = mixture_clean + rendered * gain
        interferer_meta.append((kind, placed.az_deg, placed.el_deg, sir_db))

    if n_sources:
        signal_power = np.mean([acoustic_power(direct_clean[:, c]) for c in range(n_mic)])
        noise_power = np.mean([acoustic_power(noise[:, c]) for c in range(n_mic)])
        noise_gain = float(np.sqrt(signal_power / 10 ** (layout.snr_db / 10)
                                   / (noise_power + 1e-12)))
    else:
        noise_gain = 0.05 / np.sqrt(float(np.mean(noise ** 2)) + 1e-20)
    mixture = mixture_clean + noise_gain * noise

    gain_db = rng.uniform(-args.channel_gain_db, args.channel_gain_db, size=n_mic)
    gain_db -= gain_db.mean()
    channel_gain = 10.0 ** (gain_db / 20.0)
    mixture = mixture * channel_gain[None, :]

    peak = max([float(np.max(np.abs(mixture)))]
               + [float(np.max(np.abs(s))) for s in images + directs]) + 1e-9
    wav_scale = 0.95 / peak

    def to_int16(rows):
        if not rows:
            return np.zeros((0, t_samples), dtype=np.int16)
        return np.stack([np.clip(r[:, 0] * wav_scale * 32767, -32768, 32767)
                         for r in rows]).astype(np.int16)

    directions = (np.stack([s.direction for s in layout.sources]) if layout.sources
                  else np.zeros((0, 3)))
    if n_sources > 1:
        cosines = directions @ directions.T
        np.fill_diagonal(cosines, -1.0)
        min_sep = float(np.degrees(np.arccos(np.clip(cosines.max(), -1.0, 1.0))))
    else:
        min_sep = 180.0

    meta = {
        "n_src": np.int32(n_sources),
        "d_unit": directions.astype(np.float32),
        "az_deg": np.array([s.az_deg for s in layout.sources], dtype=np.float32),
        "el_deg": np.array([s.el_deg for s in layout.sources], dtype=np.float32),
        "distance": np.array([s.distance_m for s in layout.sources], dtype=np.float32),
        "horizontal": np.array([s.horizontal_m for s in layout.sources], dtype=np.float32),
        "mouth_height": np.array([s.mouth_m for s in layout.sources], dtype=np.float32),
        "standing": np.array([s.standing for s in layout.sources], dtype=bool),
        "src_pos": (np.stack([s.position for s in layout.sources]).astype(np.float32)
                    if layout.sources else np.zeros((0, 3), dtype=np.float32)),
        "src_direct": to_int16(directs),
        "src_gain_db": np.array(gains_db, dtype=np.float32),
        "min_separation_deg": np.float32(min_sep),
        "gated": np.array(gated_flags, dtype=bool),
        "speaker_id": np.array([f"{c}:{s}" for c, s in who]),
        "interferer_kind": np.array([m[0] for m in interferer_meta]),
        "interferer_az": np.array([m[1] for m in interferer_meta], dtype=np.float32),
        "interferer_el": np.array([m[2] for m in interferer_meta], dtype=np.float32),
        "interferer_sir": np.array([m[3] for m in interferer_meta], dtype=np.float32),
        "body_center": layout.body_center.astype(np.float32),
        "room_sz": layout.room_sz.astype(np.float32),
        "mic_height_m": np.float32(layout.mic_height_m),
        "t60": np.float32(layout.t60),
        "snr": np.float32(layout.snr_db if n_sources else np.nan),
        "sensor_ratio_db": np.float32(sensor_ratio_db),
        "body_id": np.array(BODY_ID),
        "panel_indices": PANEL_INDICES,
        "panel_pos": panel_pos.astype(np.float32),
        "channel_gain_db": gain_db.astype(np.float32),
        "fs": np.int32(FS),
        "n_fft": np.int32(N_FFT_NET),
        "hop": np.int32(HOP_NET),
        "t_sec": np.float32(t_samples / FS),
        "wav_scale": np.float32(wav_scale),
    }
    if args.store_ref:
        meta["src_ref"] = to_int16(images)
    return {"wav": np.clip(mixture * wav_scale * 32767, -32768, 32767).astype(np.int16),
            "meta": meta}


def selfcheck(args) -> None:
    rng = np.random.default_rng(args.seed)
    az, el, dist, sep, mouth, mic, rooms, counts, attempts = [], [], [], [], [], [], [], [], []
    for _ in range(args.selfcheck):
        n = draw_count(rng, args.k_max, args.p_silence)
        layout = scene_mod.sample_scene(
            rng, n, args.snr_min, args.snr_max, az_limit=args.az_limit_deg,
            min_sep_deg=args.min_sep_deg,
        min_sep_az_deg=args.min_sep_az_deg, sep_median_deg=args.sep_median_deg,
            sep_sigma=args.sep_sigma)
        counts.append(n)
        attempts.append(layout.attempts)
        mic.append(layout.body_center[2])
        rooms.append(layout.room_sz)
        az += [s.az_deg for s in layout.sources]
        el += [s.el_deg for s in layout.sources]
        dist += [s.distance_m for s in layout.sources]
        mouth += [s.mouth_m for s in layout.sources]
        if n > 1:
            directions = np.stack([s.direction for s in layout.sources])
            cosines = directions @ directions.T
            np.fill_diagonal(cosines, -1.0)
            sep.append(float(np.degrees(np.arccos(np.clip(cosines.max(), -1, 1)))))

    def line(name, values, percentiles=(0, 5, 25, 50, 75, 95, 100)):
        values = np.asarray(values)
        print(f"  {name:16s} " + " ".join(
            f"p{p}={np.percentile(values, p):7.2f}" for p in percentiles))

    rooms = np.asarray(rooms)
    print(f"{args.selfcheck} clips, {len(az)} sources")
    print(f"  sources per clip {np.bincount(counts, minlength=args.k_max + 1)}"
          f"  (silent {100 * np.mean(np.array(counts) == 0):.1f} %)")
    print(f"  scene redraws    mean {np.mean(attempts):.2f}, max {max(attempts)}")
    line("azimuth [deg]", az)
    line("elevation [deg]", el)
    print(f"  {'':16s} negative {100 * np.mean(np.asarray(el) < 0):.1f} %, "
          f"outside the camera view |el|>31.8 deg {100 * np.mean(np.abs(el) > 31.8):.1f} %, "
          f"|az|>44.7 deg {100 * np.mean(np.abs(az) > 44.7):.1f} %")
    line("distance [m]", dist)
    line("mouth height [m]", mouth)
    line("mic height [m]", mic)
    line("ceiling [m]", rooms[:, 2])
    if sep:
        line("least gap [deg]", sep)
        sep = np.asarray(sep)
        for lo, hi in ((0, 10), (10, 15), (15, 22), (22, 30), (30, 180)):
            print(f"      [{lo:3d},{hi:3d}) {100 * np.mean((sep >= lo) & (sep < hi)):5.1f} %",
                  end="")
        print()


def main() -> None:
    args = parse_args()
    if args.selfcheck:
        selfcheck(args)
        return

    speech = corpora.SpeechPool(args.split)
    print(f"[{args.split}] {speech.describe()}", flush=True)
    esc_files = corpora.esc_index(args.split)
    noise_paths = corpora.noise_files()
    print(f"ESC-50 {sum(len(v) for v in esc_files.values())} clips, "
          f"{len(noise_paths)} noise files", flush=True)

    started = time.time()
    print(f"loading the BEM table ({args.bem.stat().st_size / 1e9:.2f} GB)...", flush=True)
    bem = CachedPanels(BEMTableLazy(str(args.bem), eager_load=False,
                                    apply_conjugate=RENDER_APPLY_CONJUGATE),
                       PANEL_INDICES)
    mixer = body_noise.build_mixer(bem, PANEL_INDICES, FS, N_FFT_NET,
                                   use_body=not args.free_field_noise)
    panel_pos = np.asarray(bem.centroids)[PANEL_INDICES].astype(np.float64)
    print(f"loaded in {time.time() - started:.0f}s  "
          f"diffuse noise: {'free field' if args.free_field_noise else 'through the housing'}", flush=True)

    t_samples = int(round(args.t_sec * FS))
    out_dir = args.out_dir / f"{BODY_ID}__multi" / (args.tag or args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = np.zeros(args.k_max + 1, dtype=np.int64)
    started = time.time()
    for index in range(args.start_index, args.n_samples):
        sample = synthesize(args.seed + index, bem, speech, esc_files, noise_paths,
                            mixer, panel_pos, t_samples, args)
        name = f"{index:06d}"
        sf.write(str(out_dir / f"{name}.wav"), sample["wav"], FS, subtype="PCM_16")
        np.savez(str(out_dir / f"{name}.npz"), **sample["meta"])
        counts[int(sample["meta"]["n_src"])] += 1
        if (index + 1) % 20 == 0 or index + 1 == args.n_samples:
            elapsed = time.time() - started
            done = index + 1 - args.start_index
            remaining = (args.n_samples - index - 1) * elapsed / max(done, 1)
            print(f"{index + 1}/{args.n_samples}  {elapsed:.0f}s  "
                  f"({elapsed / max(done, 1):.2f}s/clip, {remaining / 60:.0f} min left)",
                  flush=True)

    print(f"-> {out_dir}")
    for k in range(args.k_max + 1):
        print(f"   {k} sources: {counts[k]} clips")


if __name__ == "__main__":
    main()
