from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    values: dict[str, Any]

    @property
    def root_dir(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def path(self, *keys: str) -> Path:
        value: str = self.values
        for key in keys:
            value = value[key]
        return self.root_dir / value


def load_settings(config_path: Path | None = None) -> Settings:
    resolved_path = config_path or (Path(__file__).resolve().parents[2] / "config" / "config.yaml")
    with resolved_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return Settings(values=config)
