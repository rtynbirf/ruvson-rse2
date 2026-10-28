#!/usr/bin/env python3
"""RSE-2 оркестратор: Шаг 0-6 одной командой или пошагово.

Примеры:
  python run.py --config config.yaml --plan
  python run.py --config config.yaml --step 1
  python run.py --config config.yaml --step all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ruvson.config import load            # noqa: E402
from ruvson.ledger import Ledger          # noqa: E402
from ruvson import step0_okruzhenie, step1_lovi, step2_minus, step3_pasport, \
    step4_birka, step5_summa, step6_sud  # noqa: E402

STEPS = {
    "0": ("0-ОКРУЖЕНИЕ", step0_okruzhenie),
    "1": ("1-ЛОВИ", step1_lovi),
    "2": ("2-МИНУС", step2_minus),
    "3": ("3-ПАСПОРТ", step3_pasport),
    "4": ("4-БИРКА", step4_birka),
    "5": ("5-СУММА", step5_summa),
    "6": ("6-СУД", step6_sud),
}


def plan(cfg) -> None:
    print("RSE-2 план (каркас владельца, трендовый стек):")
    print("  0 ОКРУЖЕНИЕ  -> environment.json (fs/LUFS/ключ/темпо)")
    print("  1 ЛОВИ       -> src/ + manifest.json (sha256 байт-в-байт)")
    print("  2 МИНУС      -> vocals/accompaniment (roformer -> demucs fallback)")
    print("  3 ПАСПОРТ    -> passport.json + passport_f0.npz (CREPE)")
    print("  4 БИРКА      -> birka/<bank>/bank.json (референсы RUVSON)")
    print("  5 СУММА      -> vocals_ruvson + mix (openvoice|seedvc|knnvc + петля)")
    print("  6 СУД        -> judge_report.json (PASS/FAIL)")
    print(f"\nРабочая директория: {cfg.work}")
    print(f"Леджер: {cfg.work / 'ledger.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="RUVSON SINGULARITY ENGINE v2")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--step", default="all",
                    help="0..6 | all | plan")
    ap.add_argument("--plan", action="store_true",
                    help="показать план конвейера и выйти")
    args = ap.parse_args()

    cfg = load(args.config)
    ledger = Ledger(cfg.work)

    if args.plan or args.step == "plan":
        plan(cfg)
        return

    order = list(STEPS) if args.step == "all" else [args.step]
    for key in order:
        if key not in STEPS:
            raise SystemExit(f"[RUN] неизвестный шаг: {key} "
                             f"(доступны 0..6, all, plan)")
        name, mod = STEPS[key]
        print(f"\n=== {name} ===")
        mod.run(cfg, ledger)

    print(f"\nГотово. Журнал: {cfg.work / 'ledger.jsonl'}")


if __name__ == "__main__":
    main()
