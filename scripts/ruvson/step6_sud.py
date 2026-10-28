"""ШАГ 6 СУД: вердикт PASS/FAIL по приборам. Релиз только при PASS.

Судьи: Δценты (CREPE), клиппинг, LUFS, SIM (если модель доступна).
Ничего субъективного: каждая цифра в judge_report.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import judges
from .config import Config
from .ledger import Ledger


def run(cfg: Config, ledger: Ledger) -> dict:
    s = cfg["sud"]
    out_vocals = cfg.work / "sum" / "vocals_ruvson.wav"
    passport_f0 = cfg.work / "passport_f0.npz"
    if not out_vocals.exists() or not passport_f0.exists():
        raise SystemExit("[6-СУД] нет sum/vocals_ruvson.wav или паспорта — "
                         "выполни ШАГ 5 (и ШАГ 3)")

    wav, sr = judges.load_audio(out_vocals, sr=None, mono=True)
    npz = np.load(passport_f0)
    passport_track = {"times": npz["times"], "hz": npz["hz"],
                      "periodicity": npz["periodicity"]}

    track = judges.crepe_track(wav, sr)
    delta_cents = judges.drift_cents(passport_track, track)
    clip = judges.clipping_ratio(wav)
    lufs, lufs_backend = judges.measure_lufs(wav, sr)

    # SIM: если есть WeSpeaker/ECAPA-модель — считать; иначе предупреждение.
    sim, sim_note = None, "SIM-модель не подключена (Ф-4); судим по Δцентам/петле"

    reasons = []
    if not np.isfinite(delta_cents) or abs(delta_cents) > float(s["max_delta_cents"]):
        reasons.append(f"дрейф регистра {delta_cents:.1f} центов > "
                       f"{s['max_delta_cents']}")
    if clip > float(s["max_clipping_ratio"]):
        reasons.append(f"клиппинг {clip:.5f} > {s['max_clipping_ratio']}")
    lufs_dev = abs(lufs - float(s["lufs_target"]))
    if lufs_dev > float(s["lufs_tolerance"]):
        reasons.append(f"LUFS {lufs:.1f} отклоняется от {s['lufs_target']} "
                       f"на {lufs_dev:.1f} dB")
    if sim is not None and sim < float(s["min_sim"]):
        reasons.append(f"SIM {sim:.3f} < {s['min_sim']}")

    verdict = "PASS" if not reasons else "FAIL"
    report = {
        "delta_cents_median": round(delta_cents, 2) if np.isfinite(delta_cents) else None,
        "secs_sim": sim,
        "sim_note": sim_note,
        "clipping_ratio": clip,
        "lufs": round(lufs, 2),
        "lufs_backend": lufs_backend,
        "thresholds": {k: s[k] for k in ("lufs_target", "lufs_tolerance",
                                         "max_clipping_ratio", "max_delta_cents",
                                         "min_sim")},
        "verdict": verdict,
        "reasons": reasons,
    }
    (cfg.work / "judge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    ledger.log("6-СУД", "вердикт", verdict, result=report)
    print(f"\n[6-СУД] ВЕРДИКТ: {verdict}")
    for r in reasons:
        print(f"  - {r}")
    return report
