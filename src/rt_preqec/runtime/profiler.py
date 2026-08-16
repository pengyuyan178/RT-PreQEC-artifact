"""Runtime event profiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from rt_preqec.utils import ensure_parent


@dataclass
class RuntimeProfiler:
    """Collect event-level runtime records."""

    events: list[dict[str, Any]] = field(default_factory=list)

    def record_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)

    def save_csv(self, path: str | Path) -> None:
        target = ensure_parent(path)
        self.to_dataframe().to_csv(target, index=False)
