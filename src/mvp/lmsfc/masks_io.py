from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .masks import validate_masks


_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MasksFile:
    d: int
    k: int
    masks: List[int]
    threshold: Optional[int] = None
    schema_version: Optional[int] = None
    seed: Optional[int] = None
    objective: Optional[str] = None
    trainer: Optional[str] = None
    train_config: Optional[Dict[str, Any]] = None
    cost: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _coerce_int(v: Any, *, field: str) -> int:
    if isinstance(v, bool):
        raise TypeError(f"{field} must be int")
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float) and v.is_integer():
        return int(v)
    raise TypeError(f"{field} must be int")


def _coerce_float(v: Any, *, field: str) -> float:
    if isinstance(v, bool):
        raise TypeError(f"{field} must be float")
    if isinstance(v, (int, float)):
        return float(v)
    raise TypeError(f"{field} must be float")


def load_masks_json(path: str | Path) -> MasksFile:
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("masks file must be a JSON object")

    extra: Dict[str, Any] = {}
    for kk, vv in obj.items():
        if kk in {"d", "k", "masks", "threshold", "schema_version", "seed", "objective", "trainer", "train_config", "cost"}:
            continue
        extra[str(kk)] = vv

    d = _coerce_int(obj.get("d"), field="d")
    k = _coerce_int(obj.get("k"), field="k")

    masks_obj = obj.get("masks")
    if not isinstance(masks_obj, list):
        raise ValueError("masks must be a list")
    masks = [_coerce_int(v, field="masks[i]") for v in masks_obj]

    threshold_obj = obj.get("threshold", None)
    threshold: Optional[int]
    if threshold_obj is None:
        threshold = None
    else:
        threshold = _coerce_int(threshold_obj, field="threshold")

    schema_version_obj = obj.get("schema_version", None)
    schema_version: Optional[int]
    if schema_version_obj is None:
        schema_version = None
    else:
        schema_version = _coerce_int(schema_version_obj, field="schema_version")

    seed_obj = obj.get("seed", None)
    seed: Optional[int]
    if seed_obj is None:
        seed = None
    else:
        seed = _coerce_int(seed_obj, field="seed")

    objective_obj = obj.get("objective", None)
    objective: Optional[str]
    if objective_obj is None:
        objective = None
    elif isinstance(objective_obj, str):
        objective = str(objective_obj)
    else:
        raise TypeError("objective must be str")

    trainer_obj = obj.get("trainer", None)
    trainer: Optional[str]
    if trainer_obj is None:
        trainer = None
    elif isinstance(trainer_obj, str):
        trainer = str(trainer_obj)
    else:
        raise TypeError("trainer must be str")

    train_config_obj = obj.get("train_config", None)
    train_config: Optional[Dict[str, Any]]
    if train_config_obj is None:
        train_config = None
    elif isinstance(train_config_obj, dict):
        train_config = dict(train_config_obj)
    else:
        raise TypeError("train_config must be object")

    cost_obj = obj.get("cost", None)
    cost: Optional[float]
    if cost_obj is None:
        cost = None
    else:
        cost = _coerce_float(cost_obj, field="cost")

    validate_masks(masks, d)
    return MasksFile(
        d=d,
        k=k,
        masks=masks,
        threshold=threshold,
        schema_version=schema_version,
        seed=seed,
        objective=objective,
        trainer=trainer,
        train_config=train_config,
        cost=cost,
        extra=extra,
    )


def save_masks_json(
    path: str | Path,
    *,
    d: int,
    k: int,
    masks: Sequence[int],
    threshold: Optional[int] = None,
    seed: Optional[int] = None,
    objective: Optional[str] = None,
    trainer: Optional[str] = None,
    train_config: Optional[Dict[str, Any]] = None,
    cost: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    d = int(d)
    k = int(k)
    masks2 = [int(m) for m in masks]
    validate_masks(masks2, d)

    obj: Dict[str, Any] = {"schema_version": int(_SCHEMA_VERSION), "d": d, "k": k, "masks": masks2}
    if threshold is not None:
        obj["threshold"] = int(threshold)
    if seed is not None:
        obj["seed"] = int(seed)
    if objective is not None:
        obj["objective"] = str(objective)
    if trainer is not None:
        obj["trainer"] = str(trainer)
    if train_config is not None:
        obj["train_config"] = dict(train_config)
    if cost is not None:
        obj["cost"] = float(cost)
    if extra:
        for kk, vv in extra.items():
            if kk in obj:
                continue
            obj[kk] = vv

    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
