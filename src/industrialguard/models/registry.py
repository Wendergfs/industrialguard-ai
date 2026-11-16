"""Model factory."""

from __future__ import annotations

from typing import Any

from .base import BaseAnomalyModel


def list_models() -> tuple[str, ...]:
    """List supported model identifiers."""
    return ("padim", "patchcore", "efficientad")


def create_model(name: str, **kwargs: Any) -> BaseAnomalyModel:
    """Construct a configured anomaly model by registry name."""
    key = name.lower()
    if key == "padim":
        from .padim import PaDiM

        return PaDiM(**kwargs)
    if key == "patchcore":
        from .patchcore import PatchCore

        return PatchCore(**kwargs)
    if key == "efficientad":
        from .efficientad import EfficientAD

        return EfficientAD(**kwargs)
    raise ValueError(f"Unknown model {name!r}; choose one of {list_models()}")
