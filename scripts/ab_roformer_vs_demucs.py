#!/usr/bin/env python3
"""Прибор Ф-0: A/B Mel-Band RoFormer vs HTDemucs на выдержке 60-120 с
"Road to Hell" с эталонными стемами айтема archive.org.

Что меряет по каждому движку (все цифры честные, без mira_eval):
  corr_vocals  - корреляция нашего вокала с эталонным стемом Vocals
  sdr_gain     - SDR после выравнивания времени и оптимального усиления (dB)
  leakage_drums- корреляция нашего вокала с эталонными барабанами (протечка)
  sibilance    - отношение RMS полосы 6-10 кГц (вокал/эталон)
  corr_accomp  - корреляция аккомпанемента с суммой Bass+Drums+Other
  rebuild_corr - сборка (вокал+аккомп) против полного микса
Выравнивание: сдвиг по кросс-корреляции (mp3-декод даёт сдвиг до ~50 мс),
сдвиг сам попадает в отчёт - прибор не врёт молча.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, resample_poly

sys.path.insert(0, str(Path(__file__).parent))
from ruvson import judges          # noqa: E402
from ruvson.ledger import Ledger   # noqa: E402

WORK = Path("/home/z/my-project/work/rse2-run/work/road_to_hell")
SRC = WORK / "src"
AB = WORK / "abtest"
MSST = Path("/home/z/my-project/work/rse2-run/models/msst/inference.py")
CKPT = Path("/home/z/my-project/work/rse2-run/models/roformer/mel_band_roformer_vocals.ckpt")
CFGKJ = Path("/home/z/my-project/work/rse2-run/models/msst/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml")

SS, T = "60", "60"  # окно выдержки
MIX = SRC / "Chris Rea Road To Hell.HQ.mp3"
VOC = SRC / "Chris Rea Road To Hell.HQ-Vocals.mp3"
BAS = SRC / "Chris Rea Road To Hell.HQ-Bass.mp3"
DRU = SRC / "Chris Rea Road To Hell.HQ-Drums.mp3"
OTH = SRC / "Chris Rea Road To Hell.HQ-Other Instruments.mp3"


def cut(src: Path, dst: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", SS, "-t", T,
                    "-i", str(src), "-ac", "2", "-ar", "44100", str(dst)],
                   check=True, capture_output=True)


def load_mono(p: Path) -> tuple[np.ndarray, int]:
    return judges.load_audio(p, sr=None, mono=True)


def align_shift(est: np.ndarray, ref: np.ndarray, sr: int) -> int:
    """Сдвиг (в сэмплах) est относительно ref по пиковой кросс-корреляции."""
    k = 8  # даунсемпл для скорости
    a = resample_poly(est, 1, k)
    b = resample_poly(ref, 1, k)
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    c = correlate(a, b, mode="full")
    lag8 = int(np.argmax(c)) - (n - 1)
    return lag8 * k


def fit_and_score(est: np.ndarray, ref: np.ndarray, sr: int) -> dict:
    n = min(len(est), len(ref))
    e, r = est[:n].astype(np.float64), ref[:n].astype(np.float64)
    shift = align_shift(e, r, sr)
    if shift > 0:
        e = e[shift:]
        r = r[:len(r) - shift]
    elif shift < 0:
        r = r[-shift:]
        e = e[:len(e) + shift]
    n = min(len(e), len(r))
    e, r = e[:n], r[:n]
    corr = float(np.corrcoef(e, r)[0, 1])
    g = float(np.dot(r, e) / max(np.dot(e, e), 1e-12))  # оптимальное усиление
    err = r - g * e
    sdr = 10 * np.log10(max(float(np.dot(r, r)), 1e-12) /
                        max(float(np.dot(err, err)), 1e-12))
    return {"corr": round(corr, 4), "sdr_gain_db": round(sdr, 2),
            "align_shift_samples": shift}


def band_rms(w: np.ndarray, sr: int, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(w)) ** 2
    fr = np.fft.rfftfreq(len(w), 1 / sr)
    m = (fr >= lo) & (fr <= hi)
    return float(np.sqrt(spec[m].sum()))


def run_roformer() -> dict:
    din, dout = AB / "rof_in", AB / "rof_out"
    din.mkdir(parents=True, exist_ok=True)
    dout.mkdir(parents=True, exist_ok=True)
    mix = din / "mix_ex.wav"
    if not mix.exists():
        cut(MIX, mix)
    wavs = sorted(dout.rglob("*.wav"))
    if wavs:  # кэш прошлого прогона
        voc = next((p for p in wavs if "vocals" in p.name.lower()), None)
        acc = next((p for p in wavs if "instrumental" in p.name.lower()), None)
        if voc is not None:
            return {"vocals": str(voc), "accompaniment": str(acc) if acc else None,
                    "time_s": None, "files": [p.name for p in wavs],
                    "cache": "reused"}
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(MSST), "--model_type", "mel_band_roformer",
         "--config_path", str(CFGKJ), "--start_check_point", str(CKPT),
         "--input_folder", str(din), "--store_dir", str(dout),
         "--force_cpu", "--extract_instrumental"],
        capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        return {"error": proc.stderr[-600:], "cmd_time_s": round(dt, 1)}
    wavs = sorted(dout.rglob("*.wav"))
    voc = next((p for p in wavs if p.name.endswith("(vocals).wav")
                or "vocals" in p.name.lower()), None)
    acc = next((p for p in wavs if "instrumental" in p.name.lower()), None)
    return {"vocals": str(voc) if voc else None,
            "accompaniment": str(acc) if acc else None,
            "time_s": round(dt, 1), "files": [p.name for p in wavs]}


def run_demucs() -> dict:
    din, dout = AB / "dem_in", AB / "dem_out"
    din.mkdir(parents=True, exist_ok=True)
    dout.mkdir(parents=True, exist_ok=True)
    mix = din / "mix_ex.wav"
    if not mix.exists():
        cut(MIX, mix)
    sd = dout / "htdemucs" / "mix_ex"
    if (sd / "vocals.wav").exists():  # кэш прошлого прогона
        return {"vocals": str(sd / "vocals.wav"),
                "accompaniment": str(sd / "no_vocals.wav"),
                "time_s": None, "cache": "reused"}
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems", "vocals",
         "-n", "htdemucs", "-o", str(dout), str(mix)],
        capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        return {"error": proc.stderr[-600:], "cmd_time_s": round(dt, 1)}
    sd = dout / "htdemucs" / "mix_ex"
    return {"vocals": str(sd / "vocals.wav"),
            "accompaniment": str(sd / "no_vocals.wav"),
            "time_s": round(dt, 1)}


def score(engine: str, res: dict, ref_voc: np.ndarray, ref_acc: np.ndarray,
          ref_dru: np.ndarray, mix: np.ndarray, sr: int) -> dict:
    if "error" in res or not res.get("vocals"):
        return {"engine": engine, "error": res.get("error", "нет вокала")}
    ev, sre = load_mono(Path(res["vocals"]))
    out = {"engine": engine, "sep_time_s": res.get("time_s"),
           "vocals_vs_ref": fit_and_score(ev, ref_voc, sr)}
    out["leakage_drums_corr"] = round(
        fit_and_score(ev, ref_dru, sr)["corr"], 4)
    if res.get("accompaniment") and Path(res["accompaniment"]).exists():
        ea, _ = load_mono(Path(res["accompaniment"]))
        out["accomp_vs_ref"] = fit_and_score(ea, ref_acc, sr)
        n = min(len(ev), len(ea), len(mix))
        reb = ev[:n] + ea[:n]
        out["rebuild_corr"] = round(judges.correlation(mix[:n], reb), 4)
    # сибилянты: отношение RMS полосы 6-10к (после выравнивания в fit уже
    # не нужно - считаем на общих окнах без сдвига, погрешность < окна кадра)
    n = min(len(ev), len(ref_voc))
    sb = band_rms(ev[:n], sr, 6000, 10000)
    sb_ref = band_rms(ref_voc[:n], sr, 6000, 10000)
    out["sibilance_ratio_est_vs_ref"] = round(sb / max(sb_ref, 1e-12), 3)
    return out


def main() -> None:
    AB.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(WORK)
    ledger.log("Ф-0-AB", "прибор запущен", "OK",
               params={"window": f"{SS}-{int(SS) + int(T)}s",
                       "ref": "эталонные стемы айтема archive.org"})

    # эталонные стемы выдержки
    ref_voc, sr = load_mono(AB / "ref_vocals.wav") if (AB / "ref_vocals.wav").exists() \
        else (cut(VOC, AB / "ref_vocals.wav"), load_mono(AB / "ref_vocals.wav"))[1]
    for nm, src in (("ref_bass", BAS), ("ref_drums", DRU), ("ref_other", OTH)):
        p = AB / f"{nm}.wav"
        if not p.exists():
            cut(src, p)
    if not (AB / "ref_mix.wav").exists():
        cut(MIX, AB / "ref_mix.wav")

    ref_voc, sr = load_mono(AB / "ref_vocals.wav")
    rb, _ = load_mono(AB / "ref_bass.wav")
    rd, _ = load_mono(AB / "ref_drums.wav")
    ro, _ = load_mono(AB / "ref_other.wav")
    ref_acc, ref_mix = rb + rd + ro, load_mono(AB / "ref_mix.wav")[0]

    # калибровка эталона: сумма 4 стемов против микса
    cal = fit_and_score(ref_acc + ref_voc, ref_mix, sr)
    ledger.log("Ф-0-AB", "калибровка эталона (сумма стемов vs микс)", "OK",
               result=cal)

    results = []
    print("[AB] roformer: запуск (CPU, может быть долго)...", flush=True)
    r_of = run_roformer()
    print(f"[AB] roformer готов за {r_of.get('time_s')}s", flush=True)
    results.append(score("roformer_melband", r_of, ref_voc, ref_acc,
                         rd, ref_mix, sr))
    print("[AB] demucs: запуск...", flush=True)
    r_dm = run_demucs()
    print(f"[AB] demucs готов за {r_dm.get('time_s')}s", flush=True)
    results.append(score("demucs_htdemucs", r_dm, ref_voc, ref_acc,
                         rd, ref_mix, sr))

    report = {"song": "road_to_hell", "window_s": [int(SS), int(SS) + int(T)],
              "reference": "archive.org item stems (Vocals/Bass/Drums/Other)",
              "calibration_sum_stems_vs_mix": cal,
              "results": results,
              "verdict": ("roformer выигрывает по SDR" if
                          all("sdr_gain_db" in r.get("vocals_vs_ref", {})
                              for r in results if r.get("vocals_vs_ref")) and
                          results[0].get("vocals_vs_ref", {}).get("sdr_gain_db", -99) >
                          results[1].get("vocals_vs_ref", {}).get("sdr_gain_db", -99)
                          else "смотри цифры")}
    out = AB / "ab_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    ledger.log("Ф-0-AB", "вердикт A/B", "OK", result=report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
