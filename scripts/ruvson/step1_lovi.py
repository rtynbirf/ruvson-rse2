"""ШАГ 1 ЛОВИ: байт-в-байт добыча исходников + manifest.json с sha256.

Два режима (config.source):
  archive_item — качаем аудиофайлы item'а archive.org (метаданные -> файлы);
  files        — прямые url, опционально эталонные sha256.
Прецедент прогона №2: 5/5 файлов байт-в-байт.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from . import judges
from .config import Config
from .ledger import Ledger

UA = {"User-Agent": "RUVSON-RSE2/2.0 (research; contact: owner)"}


def _fetch(url: str, dest: Path) -> int:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest.stat().st_size


def _archive_item_files(item: str, formats: list[str], max_files: int) -> list[dict]:
    meta_url = f"https://archive.org/metadata/{item}"
    with urllib.request.urlopen(urllib.request.Request(meta_url, headers=UA), timeout=60) as r:
        meta = json.load(r)
    out = []
    for f in meta.get("files", []):
        name = f.get("name", "")
        ext = name.rsplit(".", 1)[-1].lower()
        if ext in formats and not name.startswith("_"):
            out.append({
                "url": f"https://archive.org/download/{item}/{urllib.request.quote(name)}",
                "path": name,
            })
    out.sort(key=lambda x: x["path"])
    return out[:max_files]


def run(cfg: Config, ledger: Ledger) -> dict:
    src = cfg.get("source", {}) or {}
    src_dir = cfg.work / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    if src.get("archive_item"):
        plan = _archive_item_files(src["archive_item"],
                                   src.get("formats", ["wav", "flac", "mp3"]),
                                   int(src.get("max_files", 5)))
    elif src.get("files"):
        plan = [{"url": f["url"], "path": Path(f["url"]).name} for f in src["files"]]
    else:
        ledger.log("1-ЛОВИ", "план добычи", "FAIL",
                   result={"error": "в config.source нет ни archive_item, ни files"})
        raise SystemExit("[1-ЛОВИ] источник не задан в config.yaml")

    manifest = {"items": []}
    ok = 0
    for item in plan:
        dest = src_dir / item["path"]
        ref_sha = None
        if "sha256" in item:
            ref_sha = item["sha256"]
        try:
            size = _fetch(item["url"], dest)
            got = judges.sha256(dest)
            match = (ref_sha == got) if ref_sha else None
            ok += int(bool(match)) if match is not None else 0
            manifest["items"].append({
                "url": item["url"], "path": str(dest), "bytes": size,
                "sha256": got, "match": match,
            })
            ledger.log("1-ЛОВИ", f"скачал {item['path']}", "OK",
                       result={"bytes": size, "sha256": got, "match": match})
        except Exception as e:
            manifest["items"].append({"url": item["url"], "path": item["path"],
                                      "error": str(e)})
            ledger.log("1-ЛОВИ", f"скачал {item['path']}", "FAIL",
                       result={"error": str(e)})

    (cfg.work / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    n_ok = sum(1 for i in manifest["items"] if "error" not in i)
    ledger.log("1-ЛОВИ", "итог добычи", "OK" if n_ok else "FAIL",
               result={"downloaded": n_ok, "total_planned": len(plan),
                       "verified_vs_reference": ok})
    return manifest
