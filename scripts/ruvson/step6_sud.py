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


def _cer_optional(cfg: Config, wav_path: Path) -> dict:
    """CER-судья по лирике: work/lyrics.txt (этикет) vs faster-whisper-транскрипт.

    Разрыв №27-линия аудита: судим разборчивость. Если лирики нет или
    faster-whisper не поставлен — честное 'not_measured', не выдумка.
    """
    lyrics = cfg.work / "lyrics.txt"
    if not lyrics.exists():
        return {"status": "not_measured", "reason": "нет work/lyrics.txt"}
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        return {"status": "not_measured",
                "reason": f"faster-whisper не поставлен ({e}) — см. 06/6.1"}
    try:
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(wav_path), language=None)
        hyp = " ".join(s.text.strip() for s in segments)
        ref = lyrics.read_text(encoding="utf-8").strip()
        import re
        ref_c = re.sub(r"\s+", "", ref.lower())
        hyp_c = re.sub(r"\s+", "", hyp.lower())
        import difflib
        ratio = difflib.SequenceMatcher(None, ref_c, hyp_c).ratio()
        cer = round((1.0 - ratio) * 100.0, 2)
        return {"status": "measured", "cer_percent": cer,
                "transcript_preview": hyp[:180]}
    except Exception as e:
        return {"status": "not_measured", "reason": str(e)[:200]}


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
    cer = _cer_optional(cfg, out_vocals)

    # SIM-судья (ECAPA, тренд 5/24): refs = sud.sim.refs, иначе voice_bank.refs.
    sim, sim_note = None, ""
    sim_refs = list((s.get("sim") or {}).get("refs") or
                    (cfg.get("voice_bank") or {}).get("refs") or [])
    if sim_refs:
        sim_res = judges.secs_sim(out_vocals, sim_refs)
        if sim_res.get("status") == "measured":
            sim = float(sim_res["sim_mean"])
            sim_note = (f"ECAPA {sim_res['model']}; sim_min={sim_res['sim_min']}; "
                        "порог калибруется в Ф-4 (ECAPA-косинус vs WavLM-SECS)")
        else:
            sim_note = f"SIM not_measured: {sim_res.get('reason')}"
    else:
        sim_note = "нет refs (sud.sim.refs / voice_bank.refs) — SIM не считаем"

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
    if cer.get("status") == "measured" and float(cer["cer_percent"]) > float(
            s.get("max_cer_percent", 20.0)):
        reasons.append(f"CER {cer['cer_percent']}% > {s.get('max_cer_percent')}")

    verdict = "PASS" if not reasons else "FAIL"
    report = {
        "delta_cents_median": round(delta_cents, 2) if np.isfinite(delta_cents) else None,
        "secs_sim": sim,
        "sim_note": sim_note,
        "sim_refs_used": bool(sim_refs),
        "clipping_ratio": clip,
        "lufs": round(lufs, 2),
        "lufs_backend": lufs_backend,
        "cer": cer,
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
