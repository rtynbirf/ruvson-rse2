"""Загрузка config.yaml — единый пульт станции."""
from pathlib import Path
import yaml


class Config:
    def __init__(self, data: dict, path: Path):
        self._d = data
        self.path = path

    def __getitem__(self, key):
        return self._d[key]

    def get(self, key, default=None):
        return self._d.get(key, default)

    @property
    def song(self) -> str:
        return self._d.get("song", "untitled")

    @property
    def work(self) -> Path:
        return Path(self._d["paths"]["work"]) / self.song

    def step_dir(self, name: str) -> Path:
        return self.work / name


def load(path: str | Path) -> Config:
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"[CONFIG] Нет файла {p}. Скопируй config.example.yaml в config.yaml и правь."
        )
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data, p)
