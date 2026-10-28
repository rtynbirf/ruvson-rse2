#!/usr/bin/env python3
"""Ф-0: склейка roformer-сегментов, контроль сборки, полный A/B против
эталонных стемов айтема (вся песня, не выдержка), продвижка в канон МИНУС.

Правило продвижки: roformer (мировой №1 на истинной разметке, +3.17 dB к
demucs) становится каноном, если контроль сборки >= 0.99. Честная запись:
на стемах ЭТОГО айтема судья смещён (стемы карaoke-происхождения), цифры
пишем как есть для обеих машин.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from ruvson import judges                    # noqa: E402
from ruvson.ledger import Ledger             # noqa: E402
from ab_roformer_vs_demucs import fit_and_score, band_rms  # noqa: E402

WORK = Path("/home/z/my-project/work/rse2-run/work/road_to_hell")
SEG = WORK / "abtest" / "seg_out"
SRC = WORK / "src"

PARTS = [(SEG / "s0" / "seg0", "seg0.wav"), (SEG / "s1" / "seg1", "seg1.wav"),
         (SEG / "s2" / "seg2", "seg2.wav")]


def concat(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts, axis=0)


def main() -> None:
    ledger = Ledger(WORK)
    # 1) склейка
    vocs, accs, sr = [], [], None
    for d, name in PARTS:
        v, s = judges.load_audio(d / "vocals.wav", sr=None, mono=True)
        a, _ = judges.load_audio(d / "instrumental.wav", sr=None, mono=True)
        vocs.append(v); accs.append(a); sr = s
    voc, acc = concat(vocs), concat(accs)
    n = min(len(voc), len(acc))
    voc, acc = voc[:n], acc[:n]

    src_wav, sr_src = judges.load_audio(
        WORK / "minus" / "demucs" / "_decoded_input.wav", sr=None, mono=True)
    m = min(n, len(src_wav))
    rebuild = judges.correlation(src_wav[:m], (voc[:m] + acc[:m]))
    ledger.log("Ф-0-ROFORMER", "склейка 3 сегментов", "OK",
               result={"length_s": round(n / sr, 2),
                       "rebuild_corr_vs_mix": round(rebuild, 6)})
    if rebuild < 0.99:
        ledger.log("Ф-0-ROFORMER", "контроль сборки", "FAIL",
                   result={"rebuild_corr": round(rebuild, 6)})
        print(f"rebuild {rebuild:.4f} < 0.99 — продвижка ОТМЕНЕНА")
        return

    # 2) полный A/B против эталонных стемов айтема (вся песня)
    refs = {}
    for key, fname in (("vocals", "Chris Rea Road To Hell.HQ-Vocals.mp3"),
                       ("bass", "Chris Rea Road To Hell.HQ-Bass.mp3"),
                       ("drums", "Chris Rea Road To Hell.HQ-Drums.mp3"),
                       ("other", "Chris Rea Road To Hell.HQ-Other Instruments.mp3")):
        p = SRC / fname
        w, s = judges.load_audio(p, sr=sr, mono=True)
        refs[key] = w[:n]
    ref_acc = refs["bass"] + refs["drums"] + refs["other"]

    dem_v, _ = judges.load_audio(WORK / "minus" / "demucs_vocals.wav"
                                 if (WORK / "minus" / "demucs_vocals.wav").exists()
                                 else WORK / "minus" / "demucs" / "htdemucs" /
                                 "_decoded_input" / "vocals.wav", sr=sr, mono=True)
    dem_a, _ = judges.load_audio(WORK / "minus" / "demucs_accompaniment.wav"
                                 if (WORK / "minus" / "demucs_accompaniment.wav").exists()
                                 else WORK / "minus" / "demucs" / "htdemucs" /
                                 "_decoded_input" / "no_vocals.wav", sr=sr, mono=True)
    dm = min(len(dem_v), len(dem_a), n)
    mix_full, _ = judges.load_audio(SRC / "Chris Rea Road To Hell.HQ.mp3",
                                    sr=sr, mono=True)
    mm = min(len(mix_full), n, dm)

    scores = {}
    for eng, v, a in (("roformer_melband", voc, acc),
                      ("demucs_htdemucs", dem_v[:dm], dem_a[:dm])):
        vs = fit_and_score(v[:n], refs["vocals"], sr)
        as_ = fit_and_score(a[:n], ref_acc, sr)
        leak = fit_and_score(v[:n], refs["drums"], sr)["corr"]
        sb = band_rms(v[:n], sr, 6000, 10000)
        sb_ref = band_rms(refs["vocals"], sr, 6000, 10000)
        reb = judges.correlation(mix_full[:mm], (v[:mm] + a[:mm]))
        scores[eng] = {"vocals_sdr_gain_db": vs["sdr_gain_db"],
                       "vocals_corr": vs["corr"],
                       "accomp_sdr_gain_db": as_["sdr_gain_db"],
                       "accomp_corr": as_["corr"],
                       "leakage_drums_corr": round(leak, 4),
                       "sibilance_ratio": round(sb / max(sb_ref, 1e-12), 3),
                       "rebuild_corr": round(reb, 6)}
        ledger.log("Ф-0-AB-ПОЛНАЯ", f"{eng} vs эталон-стемы айтема", "OK",
                   result=scores[eng])

    # 3) продвижка в канон
    minus = WORK / "minus"
    import shutil
    shutil.copyfile(minus / "vocals.wav", minus / "demucs_vocals.wav")
    shutil.copyfile(minus / "accompaniment.wav", minus / "demucs_accompaniment.wav")
    sf.write(minus / "vocals.wav", voc, sr)
    sf.write(minus / "accompaniment.wav", acc, sr)
    ledger.log("Ф-0-ROFORMER", "канон МИНУС = roformer", "OK",
               result={"note": "demucs сохранён как demucs_vocals/demucs_accompaniment",
                       "scores": scores,
                       "caution": "эталон-стемы айтема - смещённый судья; "
                                  "мир: roformer +3.17 dB на истинной разметке "
                                  "MUSDB18HQ (arXiv 2310.01809)"})
    (WORK / "abtest" / "ab_full_report.json").write_text(
        json.dumps({"calibration_note": "стемы айтема - продукт чужого "
                    "разделения (караоке-происхождение)", "scores": scores},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
