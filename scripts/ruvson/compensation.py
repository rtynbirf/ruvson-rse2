"""Петля компенсации регистра — наследие прогона №2, ядро честности тона.

OpenVoice-конвертер уводит регистр (было: -3.2 цента -> после 2 итераций
+12.6). Петля формализована:
  iter k: дрейф CREPE (конверт vs паспорт) -> коррекция в центах -> повтор
  стоп: |drift| <= tolerance ИЛИ k == max_iters
Всё пишется в compensation.json и ledger.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import judges


def _shift_wav_cents(wav: np.ndarray, sr: int, cents: float) -> np.ndarray:
    """Сдвиг высоты в центах. Точный инструмент — rubberband CLI, если есть;
    иначе librosa (качество помечаем WARN в ledger вызывающей стороны)."""
    import shutil
    if shutil.which("rubberband"):
        import soundfile as sf
        import subprocess
        tmp_in = Path("_shift_in.wav")
        tmp_out = Path("_shift_out.wav")
        sf.write(tmp_in, wav, sr)
        # rubberband принимает изменение в центах через --pitch-scale:
        subprocess.run(
            ["rubberband", "--cents", f"{int(round(cents))}",
             str(tmp_in), str(tmp_out)],
            check=True, capture_output=True)
        y, _ = sf.read(tmp_out, dtype="float32", always_2d=False)
        tmp_in.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)
        return y
    import librosa
    n_steps = cents / 100.0  # semitones, float допустим
    return librosa.effects.pitch_shift(wav, orig_sr=sr, n_steps=n_steps)


def run(compared_wav: np.ndarray, sr: int, passport_track: dict,
        convert_fn, tol_cents: float = 20.0, max_iters: int = 2,
        ledger=None) -> dict:
    """convert_fn(wav, sr) -> wav  — функция переноса тона (контура A/B/C).

    Итерация: convert -> измерить дрейф -> если |drift|>tol, сдвинуть
    конверт на -drift и пересчитать дрейф. Итог в словаре с историей.
    """
    history = []
    out = convert_fn(compared_wav, sr)
    track = judges.crepe_track(out, sr)
    drift = judges.drift_cents(passport_track, track)
    history.append({"iter": 0, "drift_cents": _f(drift), "correction_cents": 0.0})

    it = 0
    while (not np.isfinite(drift) or abs(drift) > tol_cents) and it < max_iters:
        it += 1
        correction = -drift if np.isfinite(drift) else 0.0
        out = _shift_wav_cents(out, sr, correction)
        track = judges.crepe_track(out, sr)
        drift = judges.drift_cents(passport_track, track)
        history.append({"iter": it, "drift_cents": _f(drift),
                        "correction_cents": _f(correction)})

    accepted = bool(np.isfinite(drift) and abs(drift) <= tol_cents)
    report = {
        "iterations": history,
        "final_drift_cents": _f(drift),
        "accepted": accepted,
        "tolerance_cents": tol_cents,
        "max_iters": max_iters,
    }
    if ledger:
        ledger.log("СУММА", "петля компенсации",
                   "OK" if accepted else "WARN", result=report)
    return report


def _f(x: float) -> float:
    return float(round(x, 2)) if np.isfinite(x) else float("nan")
