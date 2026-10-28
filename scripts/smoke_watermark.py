#!/usr/bin/env python3
"""Смоук AudioSeal Шага 6.5: синтетическая волна -> знак -> детекция.

Доказательство разрыва №27 (водяные знаки) исполнением, не словами.
Результат: work_smoke/ledger.jsonl + watermark.json -> evidence/.
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# 1) синтетическая "речеподобная" волна: 4 с, 16 кГц, синус с вибрато
sr = 16000
dur = 4.0
t = np.linspace(0, dur, int(sr * dur), endpoint=False)
f0 = 180.0
vibrato = f0 + 6.0 * np.sin(2 * np.pi * 5.0 * t)
phase = 2 * np.pi * np.cumsum(vibrato) / sr
tone = 0.3 * np.sin(phase) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.8 * t))

work = HERE / "work_smoke"
work.mkdir(exist_ok=True)
wav_path = work / "test_tone.wav"
with wave.open(str(wav_path), "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes((tone * 32767).astype(np.int16).tobytes())
print(f"[SMOKE] волна готова: {wav_path}")

# 2) реальный Шаг 6.5 через оркестратор
from ruvson.config import load  # noqa: E402
from ruvson.ledger import Ledger  # noqa: E402
from ruvson import watermark  # noqa: E402

cfg = load(str(HERE / "config_smoke_watermark.yaml"))
ledger = Ledger(cfg.work)
payload = watermark.run(cfg, ledger)

# 3) сводка для evidence
summary = {
    "date": "2026-10-28",
    "what": "Шаг 6.5 ЗНАК (AudioSeal) — закрытие разрыва №27",
    "model": "audioseal_wm_16bits",
    "license": "MIT (код и веса, подтверждено README 2024-04-02)",
    "input": "test_tone.wav (4 с, 16 кГц, синтетика)",
    "result": payload,
}
out = HERE / "watermark_smoke_summary.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
               encoding="utf-8")
det = payload["items"][0]
print(f"\n[SMOKE] дет-скор со знаком: {det['det_score_watermarked']} "
      f"| чистый: {det['det_score_clean']}")
print(f"[SMOKE] биты прочитаны: {det['message_decoded']}")
assert det["threshold_pass"], "дет-скор ниже 0.5 — знак не прочитан!"
print("[SMOKE] OK — разрыв №27 закрыт прибором")
