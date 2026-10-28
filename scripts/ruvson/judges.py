"""Судьи RSE-2: то, чем меряем. Всё честно, всё приборно.

Правило: судьёй регистра может быть только CREPE (torchcrepe).
librosa.yin/pyin как судьёй ЗАПРЕЩЕНЫ — октавные артефакты (приборно доказано
прогоном №2). yin-значения можно лишь предсказывать, не выносить вердикт.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_audio(path: str | Path, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    import librosa
    wav, sr = librosa.load(str(path), sr=sr, mono=mono)
    return wav, int(sr)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Корреляция двух сигналов после приведения к одной длине."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    x = a[:n] - np.mean(a[:n])
    y = b[:n] - np.mean(b[:n])
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom > 0 else 0.0


def clipping_ratio(wav: np.ndarray) -> float:
    """Доля сэмплов, прижатых к цифровому пределу (>= 0.999 full scale)."""
    if len(wav) == 0:
        return 0.0
    return float(np.mean(np.abs(wav) >= 0.999))


def measure_lufs(wav: np.ndarray, sr: int) -> tuple[float, str]:
    """Точный LUFS через pyloudnorm; без него — RMS-оценка с пометкой."""
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(wav)), "pyloudnorm"
    except Exception:
        rms = float(np.sqrt(np.mean(wav ** 2))) if len(wav) else 0.0
        est = 20 * np.log10(max(rms, 1e-9))
        return est, "rms_estimate (нет pyloudnorm — поставить по 06/6.2)"


# ------------------------------- CREPE --------------------------------------


def crepe_track(wav: np.ndarray, sr: int,
                periodicity_floor: float = 0.6,
                fmin: float = 50.0, fmax: float = 1100.0,
                model: str = "full", hop_length: int = 160,
                batch_size: int = 512) -> dict:
    """Трек f0 через torchcrepe. periodicity < floor -> NaN.

    Возвращает {"times": [...], "hz": [...], "periodicity": [...]}.
    Ничего не выдумывает: если torchcrepe не стоит — исключение с подсказкой.
    """
    try:
        import torch
        import torchcrepe
    except Exception as e:
        raise RuntimeError(
            "Нет torchcrepe/torch. СТРОГО по 06/6.2: "
            "pip install torchcrepe --index-url https://download.pytorch.org/whl/cpu "
            f"(оригинальная ошибка: {e})"
        )

    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000

    audio = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))[None]
    # ёмкость CREPE - конфигурируемая (full - стандарт оффлайна, tiny/small -
    # краевые машины; паспорт всегда пишет, каким мерилом снят)
    with torch.no_grad():
        f0, pd = torchcrepe.predict(
            audio, sr, hop_length=hop_length, fmin=fmin, fmax=fmax,
            model=model, batch_size=batch_size, device="cpu",
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)

    f0 = f0.squeeze(0).cpu().numpy()
    pd = pd.squeeze(0).cpu().numpy()
    f0[pd < periodicity_floor] = np.nan
    times = np.arange(len(f0)) * hop_length / sr
    return {"times": times, "hz": f0, "periodicity": pd}


def median_hz(track: dict) -> float:
    hz = np.asarray(track["hz"], dtype=float)
    hz = hz[~np.isnan(hz)]
    return float(np.median(hz)) if len(hz) else float("nan")


def drift_cents(track_a: dict, track_b: dict) -> float:
    """Медианный дрейф в центах: b относительно a (по медианам регистра)."""
    ma, mb = median_hz(track_a), median_hz(track_b)
    if not (ma > 0 and mb > 0):
        return float("nan")
    return float(1200 * np.log2(mb / ma))


def octave_jumps(track: dict, limit_cents: float = 700.0) -> int:
    """Число скоков > limit_cents между соседними voiced-кадрами."""
    hz = np.asarray(track["hz"], dtype=float)
    voiced = ~np.isnan(hz)
    hz_v = hz[voiced]
    if len(hz_v) < 2:
        return 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cents = 1200 * np.log2(hz_v[1:] / hz_v[:-1])
    return int(np.sum(np.abs(cents) > limit_cents))


def range_semitones(track: dict) -> float:
    hz = np.asarray(track["hz"], dtype=float)
    hz = hz[~np.isnan(hz)]
    if len(hz) < 2:
        return 0.0
    return float(12 * np.log2(np.percentile(hz, 95) / np.percentile(hz, 5)))


# ------------------------------- КЛЮЧ/ТЕМПО ---------------------------------


def key_tempo(wav: np.ndarray, sr: int) -> dict:
    """Ключ (Крумхансл-мажор/минор по хроме) и темпо (librosa). Оценки, не судьи."""
    import librosa
    chroma = librosa.feature.chroma_cqt(y=wav, sr=sr).mean(axis=1)
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    scores = []
    for i in range(12):
        c = np.roll(chroma, -i)
        scores.append((float(np.dot(c, major)), f"{pitch_classes[i]} major"))
        scores.append((float(np.dot(c, minor)), f"{pitch_classes[i]} minor"))
    best = max(scores)[1]
    onset = librosa.onset.onset_strength(y=wav, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset, sr=sr)
    return {"key_guess": best, "tempo_bpm": float(tempo[0]) if len(tempo) else float("nan")}


def secs_sim(est_path, ref_paths: list, device: str = "cpu") -> dict:
    """SIM-судья: ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb), косинус.

    Мир (аудит №1, тренд 5): порог SIM 0.74 - SECS-класс Seed-VC.
    Оговорка прибора: ECAPA-косинус не равен WavLM-SECS дословно;
    финальная калибровка порога - Ф-4 на банке RUVSON. Если модели нет -
    честное not_measured, судим по Д центам/петле (не выдумка).
    """
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception as e:
        return {"status": "not_measured",
                "reason": f"speechbrain/torchaudio не поставлены ({e})"}
    refs = [Path(p) for p in ref_paths if Path(p).exists()]
    if not refs:
        return {"status": "not_measured", "reason": "нет файлов референсов"}
    global _ECAPA
    try:
        if _ECAPA is None:
            _ECAPA = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(Path.home() / ".cache" / "ruvson-ecapa"),
                run_opts={"device": device})
    except Exception as e:
        return {"status": "not_measured",
                "reason": f"модель ECAPA не загрузилась: {str(e)[:180]}"}

    def embed(path: Path):
        # ГРАБЛЯ (приборно доказана 2026-10-28): ECAPA на длинном файле целиком
        # = OOM kill (RC 137): свёртки держат карту T×C на ВСЕХ 4.4M кадров.
        # Практика VoxCeleb: скользящее окно 10 с, шаг 5 с, средний эмбеддинг.
        wav, sr = load_audio(path, sr=None, mono=True)
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        win, hop, cap = 160000, 80000, 8  # 10 c / 5 c / <=45 c материала
        starts = list(range(0, max(len(wav) - win, 1) + 1, hop))[:cap]
        if not starts:
            starts = [0]
        embs = []
        for s0 in starts:
            t = torch.from_numpy(np.ascontiguousarray(
                wav[s0:s0 + win], dtype=np.float32))[None]
            with torch.no_grad():
                embs.append(_ECAPA.encode_batch(t).squeeze(0).squeeze(0).numpy())
        embs = np.stack(embs)
        mean = embs.mean(axis=0)
        return mean / (np.linalg.norm(mean) + 1e-12)

    import numpy as _np
    try:
        e_est = embed(Path(est_path))
        sims = [float(_np.dot(e_est, e_ref) /
                      (_np.linalg.norm(e_est) * _np.linalg.norm(e_ref) + 1e-12))
                for e_ref in (embed(r) for r in refs)]
        return {"status": "measured", "model": "speechbrain/spkrec-ecapa-voxceleb",
                "sim_mean": round(sum(sims) / len(sims), 4),
                "sim_min": round(min(sims), 4), "n_refs": len(sims)}
    except Exception as e:
        return {"status": "not_measured", "reason": f"ошибка эмбеддинга: {str(e)[:180]}"}


_ECAPA = None
