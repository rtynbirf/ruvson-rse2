"""ШАГ 6.5 ЗНАК: водяной знак релизных волн (AudioSeal, MIT — тренд №27/35).

Мировая практика (Omniseal Bench Meta): AudioSeal — локализованная детекция
до уровня сэмпла, MIT-лицензия на код И веса. Релиз без знака = тихий
уход в мир без паспорта. Деградация честная: нет пакета — FAIL в ledger
с причиной, станок не молчит.

Контракты: <work>/watermark.json (дет-скор, биты, файлы) + запись в ledger.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import Config
from .ledger import Ledger


def _candidates(cfg: Config) -> list[Path]:
    s = cfg["release"].get("watermark") or {}
    explicit = s.get("files") or []
    if explicit:
        return [Path(p) for p in explicit]
    out = []
    for rel in ("sum/mix.wav", "sum/vocals_ruvson.wav"):
        p = cfg.work / rel
        if p.exists():
            out.append(p)
    return out


def _load_models():
    from audioseal import AudioSeal  # lazy: тяжёлый импорт только здесь

    gen = AudioSeal.load_generator("audioseal_wm_16bits")
    gen.eval()
    det = AudioSeal.load_detector("audioseal_detector_16bits")
    det.eval()
    return gen, det


def watermark_file(cfg: Config, ledger: Ledger, wav_path: Path,
                   gen, det) -> dict:
    import torch

    wm = cfg["release"]["watermark"]
    sr = int(wm.get("sample_rate", 16000))
    hex_msg = str(wm.get("message_hex", "5253"))
    raw = bytes.fromhex(hex_msg)
    if len(raw) != 2:
        raise SystemExit(f"[6.5-ЗНАК] message_hex должен быть 4 hex-символа "
                         f"(16 бит), получено: {hex_msg}")
    bits = [(b >> i) & 1 for b in raw for i in range(7, -1, -1)]  # 16 x 0/1

    import soundfile as sf
    import librosa

    audio, orig_sr = librosa.load(str(wav_path), sr=sr, mono=True)
    wav_t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        watermark = gen.get_watermark(wav_t, sample_rate=sr,
                                      message=torch.tensor([bits],
                                                           dtype=torch.float32))
        watermarked = (wav_t + watermark).clamp(-1.0, 1.0)

        result, message = det.detect_watermark(watermarked)
        det_score = float(result)
        msg_out = message.detach().cpu().numpy().tolist()[0] if message is not None else None

    out_path = wav_path.with_name(wav_path.stem + "_wm.wav")
    sf.write(str(out_path),
             watermarked.squeeze(0).reshape(-1).numpy().astype(np.float32),
             sr)

    # контроль: чистый сигнал не должен давать высокую вероятность знака
    with torch.no_grad():
        result_clean, _ = det.detect_watermark(wav_t)
    det_clean = float(result_clean)

    report = {
        "engine": "audioseal (MIT, facebookresearch)",
        "source": str(wav_path.name),
        "output": out_path.name,
        "sample_rate": sr,
        "message_hex": hex_msg,
        "det_score_watermarked": round(det_score, 4),
        "det_score_clean": round(det_clean, 4),
        "message_decoded": msg_out,
        "threshold_pass": det_score >= 0.5,
    }
    status = "OK" if det_score >= 0.5 else "WARN"
    ledger.log("6.5-ЗНАК", f"watermark {wav_path.name}", status, result=report)
    return report


def run(cfg: Config, ledger: Ledger) -> dict | None:
    wm_cfg = cfg["release"].get("watermark") or {}
    if not wm_cfg.get("enabled", False):
        rec = {"reason": "release.watermark.enabled=false — знак выключен в конфиге"}
        ledger.log("6.5-ЗНАК", "watermark", "INFO", result=rec)
        print("[6.5-ЗНАК] отключён в конфиге (релизы без знака) — запись в ledger есть")
        return None

    files = _candidates(cfg)
    if not files:
        raise SystemExit("[6.5-ЗНАК] нет релизных волн (sum/mix.wav | "
                         "sum/vocals_ruvson.wav | release.watermark.files)")

    try:
        gen, det = _load_models()
    except Exception as e:
        ledger.log("6.5-ЗНАК", "watermark", "FAIL",
                   result={"reason": f"audioseal недоступен: {e}", "fix": "pip install audioseal (см. 06/6.7)"})
        raise SystemExit("[6.5-ЗНАК] audioseal не поставлен — см. 06_РЕЕСТР/6.7")

    reports = []
    for p in files:
        reports.append(watermark_file(cfg, ledger, p, gen, det))

    payload = {"model": "audioseal_wm_16bits", "license": "MIT (код и веса)",
               "items": reports}
    (cfg.work / "watermark.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = all(r["threshold_pass"] for r in reports)
    print(f"\n[6.5-ЗНАК] {'ПРОЙДЕН' if ok else 'С ТРЕВОГОЙ'}: "
          f"{len(reports)} файл(ов), watermark.json записан")
    return payload
