from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def default_config_path() -> Path:
    here = Path(__file__).resolve()
    repo = here.parents[1]
    candidate = repo / "configs" / "default.yaml"
    if candidate.exists():
        return candidate
    packaged = here.parent / "configs" / "default.yaml"
    return packaged


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else default_config_path()
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if overrides:
        data = _deep_update(data, overrides)
    return data


def flatten_overrides(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    """Turn dotted CLI keys such as detector.conf=0.4 into a nested dict."""
    root: dict[str, Any] = {}
    for dotted, value in pairs:
        keys = dotted.split(".")
        cursor = root
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = value
    return root


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> "Config":
        return cls(load_config(path, overrides))

    def get(self, *keys: str, default: Any = None) -> Any:
        cursor: Any = self.raw
        for key in keys:
            if not isinstance(cursor, dict) or key not in cursor:
                return default
            cursor = cursor[key]
        return cursor

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        return dict(value) if isinstance(value, dict) else {}
