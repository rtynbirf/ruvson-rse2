"""ШАГ 5 СУММА: перенос тембра RUVSON на вокал песни + сборка СУММА++.

Три контура (config.sum.engine):
  openvoice — OpenVoice v2 tone converter (MIT, доказан) + ПЕТЛЯ компенсации;
  seedvc    — Seed-VC zero-shot SVC 44.1k (GPL-3, ТОЛЬКО внешний CLI-процесс);
  knnvc     — kNN-VC retrieval без обучения (MIT) — аварийный контур.
Сборка: vocals_ruvson + accompaniment -> sum/mix.wav (АНСАМБЛЬ).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from . import judges, compensation
from .config import Config
from .ledger import Ledger


# ------------------------------ контур B: OpenVoice --------------------------

def _convert_openvoice(cfg: Config):
    """Возвращает convert_fn(wav, sr) -> wav. Чекпойнт: converter/checkpoint.pth."""
    oc = cfg["sum"]["openvoice"]
    ckpt = Path(oc["converter_ckpt"])
    cconf = Path(oc["converter_config"])
    if not ckpt.exists() or not cconf.exists():
        raise RuntimeError(
            f"нет чекпойнта OpenVoice: {ckpt} / {cconf} — см. 06/6.2 п.4 "
            "(именно converter/checkpoint.pth 131 МБ, не checkpoint_v2)")

    try:
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter
    except Exception as e:
        raise RuntimeError(
            f"пакет openvoice не импортируется ({e}) — поставить по 06/6.2")

    tau = float(oc.get("tau", 0.3))
    device = "cpu"
    converter = ToneColorConverter(str(cconf), device=device)
    converter.load_ckpt(str(ckpt))

    bank_refs = [Path(p) for p in (cfg["voice_bank"].get("refs") or [])
                 if Path(p).exists()]
    if not bank_refs:
        raise RuntimeError("БИРКА пуста: нет референсов для тона RUVSON")
    tmp_ref = Path(cfg.work / "birka" / "_ref_for_tone.wav")
    if not tmp_ref.exists():
        wav, sr = judges.load_audio(bank_refs[0], sr=None, mono=True)
        sf.write(tmp_ref, wav, sr)

    ref_se, _ = se_extractor.get_se(tmp_ref, converter, target_dir=str(tmp_ref.parent),
                                    vad=False)

    def convert(wav: np.ndarray, sr: int) -> np.ndarray:
        tmp_in = cfg.work / "sum" / "_ov_in.wav"
        tmp_out = cfg.work / "sum" / "_ov_out.wav"
        sf.write(tmp_in, wav, sr)
        converter.convert(audio_src_path=str(tmp_in), src_se=None,
                          tgt_se=ref_se, output_path=str(tmp_out), tau=tau,
                          message="@RUVSON")
        out, _ = sf.read(tmp_out, dtype="float32", always_2d=False)
        if out.ndim > 1:
            out = out.mean(axis=1)
        tmp_in.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)
        return out

    return convert


# ------------------------------ контур A: Seed-VC ----------------------------

def _convert_seedvc(cfg: Config):
    """GPL-изоляция: только subprocess в чужом репо, без импорта."""
    sc = cfg["sum"]["seedvc"]
    repo = Path(sc["repo_dir"])
    if not repo.exists():
        raise RuntimeError(
            "нет models/seed-vc — клонируй https://github.com/Plachtaa/seed-vc "
            "в отдельное окружение (GPL-3), запуск только CLI")
    ref = next((Path(p) for p in (cfg["voice_bank"].get("refs") or [])
                if Path(p).exists()), None)
    if ref is None:
        raise RuntimeError("нет референса для Seed-VC (voice_bank.refs)")

    def convert(wav: np.ndarray, sr: int) -> np.ndarray:
        tmp_in = cfg.work / "sum" / "_seedvc_in.wav"
        tmp_out_dir = cfg.work / "sum"
        sf.write(tmp_in, wav, sr)
        cmd = [sc.get("python", sys.executable), str(repo / "inference.py"),
               "--source", str(tmp_in), "--target", str(ref),
               "--output", str(tmp_out_dir), "--diffusion_steps", "30"]
        if sc.get("checkpoint"):
            cmd += ["--checkpoint", str(sc["checkpoint"]),
                    "--config", str(sc["config"])]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"seed-vc CLI упал: {proc.stderr[-600:]}")
        out_path = sorted(tmp_out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)[-1]
        out, _ = sf.read(out_path, dtype="float32", always_2d=False)
        if out.ndim > 1:
            out = out.mean(axis=1)
        return out

    return convert


# ------------------------------ контур C: kNN-VC -----------------------------

def _convert_knnvc(cfg: Config):
    raise RuntimeError(
        "контур kNN-VC подключается в Ф-3 (см. 02_дорожная карта); "
        "стенд: github.com/bshall/knn-vc, веса в models/knnvc")


CONVERTERS = {"openvoice": _convert_openvoice,
              "seedvc": _convert_seedvc,
              "knnvc": _convert_knnvc}


def run(cfg: Config, ledger: Ledger) -> dict:
    vocals = cfg.work / "minus" / "vocals.wav"
    passport_f0 = cfg.work / "passport_f0.npz"
    if not vocals.exists() or not passport_f0.exists():
        raise SystemExit("[5-СУММА] нет minus/vocals.wav или passport_f0.npz — "
                         "выполни ШАГ 2 и ШАГ 3")

    engine = cfg["sum"].get("engine", "openvoice")
    (cfg.work / "sum").mkdir(parents=True, exist_ok=True)

    wav, sr = judges.load_audio(vocals, sr=None, mono=True)
    npz = np.load(passport_f0)
    passport_track = {"times": npz["times"], "hz": npz["hz"],
                      "periodicity": npz["periodicity"]}

    try:
        convert_fn = CONVERTERS[engine](cfg)
    except RuntimeError as e:
        ledger.log("5-СУММА", f"контур {engine}", "FAIL",
                   result={"error": str(e)})
        raise SystemExit(f"[5-СУММА] {e}")

    # Петля компенсации: convert -> дрейф -> коррекция -> ... (<= max_iters)
    comp_cfg = cfg["sum"]["compensation"]
    report = compensation.run(
        wav, sr, passport_track, convert_fn,
        tol_cents=float(comp_cfg.get("tolerance_cents", 20.0)),
        max_iters=int(comp_cfg.get("max_iters", 2)),
        ledger=ledger,
    )
    if any(i.get("correction_cents") for i in report["iterations"][1:]):
        ledger.log("5-СУММА", "коррекция сдвигом", "WARN",
                   result={"note": "применён pitch-shift корректор; для максимума "
                                   "качества поставить rubberband CLI"})

    converted = convert_fn(wav, sr)
    out_vocals = cfg.work / "sum" / "vocals_ruvson.wav"
    sf.write(out_vocals, converted, sr)

    # СБОРКА СУММА++: голос RUVSON + аккомпанемент = АНСАМБЛЬ
    acc_path = cfg.work / "minus" / "accompaniment.wav"
    mix_path = cfg.work / "sum" / "mix.wav"
    if acc_path.exists():
        acc, sr_a = judges.load_audio(acc_path, sr=sr, mono=True)
        # LUFS-match (аудит №1, коррекция 4): голос подгоняется к нише,
        # не наоборот. Практика пост-цепи каверов: ±2 dB.
        match = bool(cfg["sum"]["mix"].get("lufs_match", True))
        n = min(len(converted), len(acc))
        voc_mix = converted[:n]
        acc_mix = acc[:n]
        if match:
            lufs_v, _ = judges.measure_lufs(voc_mix, sr)
            lufs_a, _ = judges.measure_lufs(acc_mix, sr)
            if np.isfinite(lufs_v) and np.isfinite(lufs_a):
                gain_db = float(np.clip(lufs_a - lufs_v, -12.0, 12.0))
                voc_mix = voc_mix * (10 ** (gain_db / 20))
                ledger.log("5-СУММА", "LUFS-match голоса к нише", "OK",
                           result={"gain_db": round(gain_db, 2),
                                   "lufs_vocals": round(lufs_v, 2),
                                   "lufs_accomp": round(lufs_a, 2)})
        mix = voc_mix + acc_mix
        peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
        limit = 10 ** (float(cfg["sum"]["mix"].get("peak_dbfs", -1.0)) / 20)
        if peak > limit:
            mix = mix * (limit / peak)
        sf.write(mix_path, mix, sr)
        ledger.log("5-СУММА", "сборка ансамбля", "OK",
                   result={"mix": str(mix_path), "peak_dbfs": round(20 * np.log10(max(peak, 1e-9)), 2)})
    else:
        ledger.log("5-СУММА", "сборка ансамбля", "WARN",
                   result={"note": "нет аккомпанемента — микс не собран"})

    (cfg.work / "sum" / "compensation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"engine": engine, "vocals_ruvson": str(out_vocals),
            "compensation": report}
