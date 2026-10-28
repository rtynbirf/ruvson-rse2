"""Единый журнал действий: ledger.jsonl.

Тишины не бывает: успех ИЛИ отказ с причиной — всё в журнал.
"""
import json
import time
from pathlib import Path


class Ledger:
    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.work_dir / "ledger.jsonl"

    def log(self, step: str, action: str, status: str,
            params: dict | None = None, result: dict | None = None):
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "step": step,
            "action": action,
            "status": status,          # OK | FAIL | WARN | INFO
            "params": params or {},
            "result": result or {},
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tag = f"[{step}]"
        print(f"{tag} {action} -> {status}"
              + (f" | {json.dumps(result, ensure_ascii=False)}" if result else ""))
        return rec
