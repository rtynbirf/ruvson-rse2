"""ШАГ 0 ОКРУЖЕНИЕ:_fs/каналы/длительность/LUFS/ключ/темпо источника.

ОБЪЕКТ БЕЗ ОКРУЖЕНИЯ = ОШИБКА: без environment.json следующие шаги не стартуют.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import judges
from .config import Config
from .ledger import Ledger


def run(cfg: Config, ledger: Ledger) -> dict:
    src_dir = cfg.work / "src"
    files = sorted([p for p in src_dir.glob("*") if p.suffix.lower()
                    in {".wav", ".flac", ".mp3"}]) if src_dir.exists() else []

    if not files:
        res = {"error": "нет исходников в work/<song>/src — сначала ШАГ 1 ЛОВИ"}
        ledger.log("0-ОКРУЖЕНИЕ", "инвентаризация", "FAIL", result=res)
        raise SystemExit(f"[0-ОКРУЖЕНИЕ] {res['error']}")

    # Профиль окружения строится от ПОЛНОГО МИКСА, не от стема:
    # подсказка source.primary_hint (та же дисциплина, что в ШАГ 2 МИНУС).
    # Коррекция 2026-10-28 (Ф-0, ревизия №2): без неё первоисточником
    # становился первый по алфавиту стем (Bass) — LUFS/ключ/темпо врали.
    hint = (cfg.get("source") or {}).get("primary_hint")
    primary = next((p for p in files if hint and hint.lower() in p.name.lower()),
                   files[0])
    if hint and hint.lower() not in primary.name.lower():
        ledger.log("0-ОКРУЖЕНИЕ", "первоисточник", "WARN",
                   result={"note": "primary_hint не совпал ни с одним файлом — "
                                   "профиль по первому файлу, задай подсказку!"})
    wav, sr = judges.load_audio(primary, sr=None, mono=True)
    lufs, lufs_backend = judges.measure_lufs(wav, sr)
    kt = judges.key_tempo(wav, sr)

    env = {
        "primary_source": str(primary),
        "files_found": [str(p.name) for p in files],
        "sample_rate": sr,
        "channels": 1,  # анализ по моно-миксу; исходное число каналов из manifest
        "duration_s": round(len(wav) / sr, 3),
        "lufs": round(lufs, 2),
        "lufs_backend": lufs_backend,
        **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in kt.items()},
    }

    out = cfg.work / "environment.json"
    out.write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    ledger.log("0-ОКРУЖЕНИЕ", "профиль окружения", "OK", result=env)
    return env
