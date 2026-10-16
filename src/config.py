"""Loads configs/*.yaml into a nested, dot-accessible config object.

Keeping this tiny on purpose: every pipeline stage reads its parameters from
here so nothing is hardcoded in preprocessing.py / tfd_transform.py / model.py.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw)
