"""ШАГ 4 БИРКА: банк голоса RUVSON — единственный источник тембра.

Банк = референсы живого вокала + статистика регистра + место под
эмбеддинги (ECAPA/WeSpeaker) и kNN-фичи (контур C). Контракт bank.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import judges
from .config import Config
from .ledger import Ledger


def run(cfg: Config, ledger: Ledger) -> dict:
    vb = cfg["voice_bank"]
    refs = [Path(p) for p in (vb.get("refs") or [])]
    refs = [p for p in refs if p.exists()]
    if not refs:
        res = {"error": "в voice_bank.refs нет ни одного существующего WAV. "
                        "Референсы живого вокала RUVSON обязательны."}
        ledger.log("4-БИРКА", "сбор банка", "FAIL", result=res)
        raise SystemExit(f"[4-БИРКА] {res['error']}")

    bank_dir = cfg.work / "birka" / vb.get("name", "bank")
    bank_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for p in refs:
        wav, sr = judges.load_audio(p, sr=None, mono=True)
        entry = {"path": str(p), "seconds": round(len(wav) / sr, 2),
                 "sample_rate": sr}
        try:
            track = judges.crepe_track(wav, sr)
            entry["median_hz"] = round(judges.median_hz(track), 3)
            entry["range_semitones"] = round(judges.range_semitones(track), 2)
        except RuntimeError as e:
            entry["f0"] = f"skipped: {e}"
            ledger.log("4-БИРКА", f"f0 референса {p.name}", "WARN",
                       result={"reason": str(e)[:200]})
        entries.append(entry)
        ledger.log("4-БИРКА", f"референс {p.name}", "OK",
                   result={k: entry[k] for k in
                           ("seconds", "sample_rate", "median_hz") if k in entry})

    bank = {
        "name": vb.get("name", "bank"),
        "created": json.dumps(json.__class__ and __import__("datetime").datetime
                              .now().isoformat(timespec="seconds")),
        "refs": entries,
        "embeddings": {"model": None, "dim": None,
                       "note": "ECAPA/WeSpeaker подключается в Ф-4 по 02/дорожной карте"},
        "knn_features": {"model": None,
                         "note": "kNN-VC фичи — контур C, Ф-3"},
        "tone_stats": {
            "median_hz_all": (round(float(np.median([e["median_hz"] for e in entries
                                                     if "median_hz" in e])), 3)
                              if any("median_hz" in e for e in entries) else None),
        },
    }
    (bank_dir / "bank.json").write_text(
        json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")
    ledger.log("4-БИРКА", "банк собран", "OK",
               result={"bank": str(bank_dir / "bank.json"),
                       "refs": len(entries)})
    return bank
