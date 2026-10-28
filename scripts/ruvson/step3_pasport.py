"""ШАГ 3 ПАСПОРТ песни: трек f0 (CREPE), регистр, ключ, темпо, LUFS.

Один судья — CREPE. Ноль октавных скачков = паспорт годен.
Выход: passport.json — обязательный вход для БИРКИ/СУММА/СУД.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import judges
from .config import Config
from .ledger import Ledger


def run(cfg: Config, ledger: Ledger) -> dict:
    vocals = cfg.work / "minus" / "vocals.wav"
    if not vocals.exists():
        raise SystemExit("[3-ПАСПОРТ] нет minus/vocals.wav — сначала ШАГ 2")

    crepe_cfg = cfg["passport"]["crepe"]
    wav, sr = judges.load_audio(vocals, sr=None, mono=True)

    track = judges.crepe_track(
        wav, sr,
        periodicity_floor=crepe_cfg.get("periodicity_floor", 0.6),
        fmin=crepe_cfg.get("fmin", 50.0),
        fmax=crepe_cfg.get("fmax", 1100.0),
        model=crepe_cfg.get("model", "full"),
        hop_length=int(crepe_cfg.get("hop_length", 160)),
        batch_size=int(crepe_cfg.get("batch_size", 512)),
    )
    jumps = judges.octave_jumps(track, crepe_cfg.get("octave_jump_cents", 700.0))
    lufs, lufs_backend = judges.measure_lufs(wav, sr)
    kt = judges.key_tempo(wav, sr)

    passport = {
        "source": str(vocals),
        "f0_engine": f"torchcrepe/CREPE {crepe_cfg.get('model', 'full')}",
        "f0_summary": {
            "median_hz": round(judges.median_hz(track), 3),
            "range_semitones": round(judges.range_semitones(track), 2),
            "octave_jumps": jumps,
        },
        "lufs": round(lufs, 2),
        "lufs_backend": lufs_backend,
        **kt,
        "crepe_params": crepe_cfg,
    }

    # Полный трек сохраняем рядом (npz), в passport.json — только сводка
    np.savez_compressed(cfg.work / "passport_f0.npz",
                        times=track["times"], hz=track["hz"],
                        periodicity=track["periodicity"])
    (cfg.work / "passport.json").write_text(
        json.dumps(passport, ensure_ascii=False, indent=2), encoding="utf-8")

    status = "OK" if jumps == 0 else "WARN"
    ledger.log("3-ПАСПОРТ", "паспорт песни", status, result=passport["f0_summary"]
               | {"lufs": passport["lufs"], "key": kt["key_guess"],
                  "tempo": kt["tempo_bpm"]})
    if jumps:
        ledger.log("3-ПАСПОРТ", "октавные скачки", "WARN",
                   result={"count": jumps,
                           "note": "проверь стем вокала на артефакты разделения"})
    return passport
