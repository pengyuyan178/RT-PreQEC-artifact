"""Sinter integration placeholder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SinterRunResult:
    """Placeholder container for sinter Monte Carlo outputs."""

    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def run_sinter_sampling(*_: Any, **__: Any) -> SinterRunResult:
    """Return a placeholder until full sinter integration is added."""
    return SinterRunResult(success=False, metadata={"placeholder": True})
