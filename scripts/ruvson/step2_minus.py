"""ШАГ 2 МИНУС: вычесть инструмент (голос) из окружения.

Движки:
  roformer — Mel-Band RoFormer (vocals SDR 11.21 dB, мировой №1 SDX'23);
             запускается MSST-совместимым внешним скриптом при наличии весов.
  demucs   — htdemucs (8.04 dB), MIT, доказан прогоном №2 (0.9992).
  auto     — roformer если есть веса, иначе demucs (с записью в ledger).
После разделения — контроль сборки: vocals+accompaniment vs источник.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import judges
from .config import Config
from .ledger import Ledger


def _decode_to_wav(src: Path, out_wav: Path, sr: int) -> dict:
    """Принудительный ffmpeg-преддекод в WAV.

    ГРАБЛЯ (приборно доказана 2026-10-28): demucs 4.1 грузит аудио через
    sphn.read (Rust-декодер), и на этом mp3 он выдаёт мусор (corr 0.0026
    с ffmpeg-эталоном) — разделение получает не песню. Поэтому станок
    декодирует САМ, детерминированным декодером, и отдаёт demucs WAV.
    """
    import shutil as _sh
    if not _sh.which("ffmpeg"):
        return {"error": "ffmpeg не найден в PATH — обязателен для декод-контроля"}
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ac", "2", "-ar", str(sr), str(out_wav)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        return {"error": f"ffmpeg упал: {proc.stderr[-300:]}"}
    import soundfile as sf
    import numpy as _np
    wav, _sr = sf.read(out_wav, dtype="float32", always_2d=False)
    rms = float(_np.sqrt((wav ** 2).mean()))
    if rms < 1e-4:
        return {"error": f"декод-контроль: WAV почти пуст (rms={rms:.6f})"}
    return {"wav": str(out_wav), "rms": round(rms, 5),
            "note": "декод-контроль пройден"}


def _run_demucs(cfg: Config, src: Path, out_root: Path, model: str) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    decoded = _decode_to_wav(src, out_root / "_decoded_input.wav",
                             int(cfg["minus"].get("sample_rate", 44100)))
    if "error" in decoded:
        return decoded
    feed = Path(decoded["wav"])
    cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals",
           "-n", model, "-o", str(out_root), str(feed)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"error": f"demucs вернул {proc.returncode}: {proc.stderr[-800:]}",
                "cmd": cmd}
    song_dir = out_root / model / feed.stem
    vocals = song_dir / "vocals.wav"
    accomp = song_dir / "no_vocals.wav"
    if not vocals.exists():
        return {"error": f"нет {vocals} после demucs", "cmd": cmd}
    return {"vocals": str(vocals), "accompaniment": str(accomp),
            "engine": "demucs", "decode": decoded["note"]}


def _run_samaudio(cfg: Config, src: Path, out_root: Path) -> dict:
    """SAM-Audio (Meta): разделение по текстовой подсказке — тренд №4.

    Стенд не подключён — честный отказ, станок не молчит. Точка входа
    зарезервирована: models/samaudio, prompt из config minus.samaudio.prompt.
    """
    sa = (cfg["minus"].get("samaudio") or {})
    weights = Path(sa.get("weights_dir", "./models/samaudio"))
    if not weights.exists() or not any(weights.glob("*")):
        return {"error": "SAM-Audio стенд не подключён (models/samaudio пуст). "
                         "Разрыв №4 аудита №1: мир разделяет по тексту; включаем "
                         "после выхода весов, сейчас честный откат на roformer/demucs"}
    return {"error": "SAM-Audio веса найдены, но точка входа ещё не написана — см. 02 Ф-RT"}


def _run_roformer(cfg: Config, src: Path, out_root: Path) -> dict:
    r = cfg["minus"]["roformer"]
    weights = Path(r.get("weights_dir", ""))
    if not weights.exists() or not any(weights.glob("*")):
        return {"error": "веса roformer не найдены (minus.roformer.weights_dir пуст)"}
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = [r.get("python", sys.executable), r.get("entry", "msst_infer.py"),
           "--model_type", "mel_band_roformer", "--config_path", r.get("config", ""),
           "--start_check_point", str(weights / "mel_band_roformer_vocals.ckpt"),
           "--input_folder", str(src.parent), "--store_dir", str(out_root)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"error": f"roformer вернул {proc.returncode}: {proc.stderr[-800:]}",
                "cmd": cmd}
    vocals = next((p for p in out_root.rglob("*vocals*.wav")), None)
    if vocals is None:
        return {"error": "roformer не выдал vocals-стем"}
    return {"vocals": str(vocals), "engine": "roformer",
            "note": "аккомпанемент = src - vocals; для СУММА++ использовать оба стема"}


def run(cfg: Config, ledger: Ledger) -> dict:
    src_dir = cfg.work / "src"
    files = sorted(p for p in src_dir.glob("*")
                   if p.suffix.lower() in {".wav", ".flac", ".mp3"})
    if not files:
        raise SystemExit("[2-МИНУС] нет src-файлов — сначала ШАГ 1")
    # Первоисточник = ПОЛНЫЙ МИКС. Подсказка из config.source.primary_hint
    # защищает от выбора стема (Bass/Drums/Vocals) как входа разделения.
    hint = (cfg.get("source") or {}).get("primary_hint")
    primary = next((p for p in files if hint and hint.lower() in p.name.lower()),
                   files[0])
    ledger.log("2-МИНУС", "первоисточник", "OK",
               result={"primary": primary.name,
                       "note": "полный микс, не стем" if hint else
                               "по умолчанию первый файл — задай source.primary_hint!"})

    out_root = cfg.work / "minus"
    engine = cfg["minus"].get("engine", "auto")
    res = None
    tried = []

    order = {"auto": ["roformer", "demucs"],
             "roformer": ["roformer"],
             "samaudio": ["samaudio", "roformer", "demucs"],
             "demucs": ["demucs"]}[engine]
    for eng in order:
        tried.append(eng)
        if eng == "roformer":
            res = _run_roformer(cfg, primary, out_root / eng)
        elif eng == "samaudio":
            res = _run_samaudio(cfg, primary, out_root / eng)
        else:
            res = _run_demucs(cfg, primary, out_root / eng,
                              cfg["minus"].get("demucs_model", "htdemucs"))
        if "error" not in res:
            ledger.log("2-МИНУС", f"движок {eng}", "OK",
                       params={"engine": eng}, result=res)
            break
        ledger.log("2-МИНУС", f"движок {eng}", "FAIL",
                   params={"engine": eng}, result={"error": res["error"]})
    else:
        raise SystemExit("[2-МИНУС] ни один движок не сработал")

    # Контроль сборки (наследие 0.9992): vocals + accompaniment ≈ источник
    vocals, sr_v = judges.load_audio(res["vocals"], sr=None, mono=True)
    acc_path = res.get("accompaniment")
    corr = None
    if acc_path and Path(acc_path).exists():
        acc, sr_a = judges.load_audio(acc_path, sr=sr_v, mono=True)
        n = min(len(vocals), len(acc))
        rebuilt = vocals[:n] + acc[:n]
        src_wav, _ = judges.load_audio(primary, sr=sr_v, mono=True)
        corr = judges.correlation(src_wav[:len(rebuilt)], rebuilt)
    res["rebuild_correlation"] = round(corr, 6) if corr is not None else None

    if corr is not None and corr < 0.99:
        ledger.log("2-МИНУС", "контроль сборки", "WARN",
                   result={"correlation": res["rebuild_correlation"],
                           "порог": 0.99,
                           "note": "сборка хуже порога — проверь нормализацию стемов"})

    # Копия стемов в minus/ — канонические пути для следующих шагов
    import shutil
    can_vocals = cfg.work / "minus" / "vocals.wav"
    can_accomp = cfg.work / "minus" / "accompaniment.wav"
    shutil.copyfile(res["vocals"], can_vocals)
    if acc_path and Path(acc_path).exists():
        shutil.copyfile(acc_path, can_accomp)
    res["canonical_vocals"] = str(can_vocals)
    res["canonical_accompaniment"] = str(can_accomp) if acc_path else None

    (cfg.work / "minus_report.json").write_text(
        json.dumps({"tried": tried, **{k: v for k, v in res.items()}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    ledger.log("2-МИНУС", "контроль сборки", "OK",
               result={"rebuild_correlation": res["rebuild_correlation"]})
    return res
