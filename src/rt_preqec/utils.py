"""General utility helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_parent(path: str | Path) -> Path:
    """Ensure a path's parent directory exists."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def to_jsonable(data: dict[str, Any]) -> dict[str, Any]:
    """Convert non-serializable values into basic JSON-friendly types."""
    converted: dict[str, Any] = {}
    for key, value in data.items():
        if hasattr(value, "tolist"):
            converted[key] = value.tolist()
        elif isinstance(value, dict):
            converted[key] = to_jsonable(value)
        else:
            converted[key] = value
    return converted


def dump_json(data: dict[str, Any], path: str | Path) -> None:
    """Write JSON with indentation."""
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2)
