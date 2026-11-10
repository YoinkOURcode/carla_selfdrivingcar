"""Fixed-shape observation packing for DreamerV3 attention inputs.

Coordinates are ego vehicle coordinates: x forward, y right, z up. Distances
are metres and speeds are metres/second. Object and context feature meanings
are defined by the environment; each token has 16 float32 values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass
class FrameBundle:
    """Measurements associated with one simulator frame.

    ``rgb`` is HWC RGB uint8; depth is HxW metres (or None). Radar values are
    ego-frame arrays with xyz metres and radial velocity m/s in the first four
    columns. ``valid`` distinguishes an available empty scan from a missing one.
    """

    frame: int
    timestamp: float
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    radars: Mapping[str, np.ndarray] = field(default_factory=dict)
    valid: Mapping[str, bool] = field(default_factory=dict)
    calibration: Mapping[str, Any] = field(default_factory=dict)
    events: tuple[Mapping[str, Any], ...] = ()


def _resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize; size is (width, height)."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape [height, width, 3]")
    if image.dtype != np.uint8:
        raise ValueError("rgb must use uint8 values")
    width, height = size
    if width < 1 or height < 1:
        raise ValueError("model_size dimensions must be positive")
    ys = np.minimum((np.arange(height) * image.shape[0] / height).astype(int), image.shape[0] - 1)
    xs = np.minimum((np.arange(width) * image.shape[1] / width).astype(int), image.shape[1] - 1)
    return image[ys[:, None], xs[None, :], :]


def pack_observation(
    rgb: np.ndarray | None,
    context: np.ndarray,
    objects: np.ndarray | None = None,
    radars: Mapping[str, np.ndarray] | None = None,
    radar_valid: Mapping[str, bool] | None = None,
    *,
    max_objects: int = 32,
    radar_capacity: int = 8,
    model_size: tuple[int, int] = (64, 64),
) -> dict[str, np.ndarray]:
    """Pack modalities into fixed-shape arrays suitable for TensorDict.

    Token layout: context (1), objects (``max_objects``), and four radar
    groups in front/left/right/rear order (``radar_capacity`` each). Missing
    slots are zero and have mask=False. RGB absence produces a black image;
    callers should use the optional ``pixels_valid`` field to distinguish it.
    Each radar input row starts with ego x/y/z in metres and radial velocity
    in m/s; those channels are normalized by 100m and 30m/s. Additional
    channels are retained without scaling. Radar rows are nearest-first and
    duplicates within 0.5m are suppressed. ``radar_valid`` distinguishes a
    valid empty scan from a missing sensor; if omitted, keys in ``radars`` are
    treated as valid.
    """
    if max_objects < 0 or radar_capacity < 0:
        raise ValueError("capacities cannot be negative")
    context = np.asarray(context, dtype=np.float32)
    if context.shape != (16,):
        raise ValueError("context must have shape [16]")
    total = 1 + max_objects + 4 * radar_capacity
    tokens = np.zeros((total, 16), dtype=np.float32)
    mask = np.zeros(total, dtype=np.bool_)
    if np.isfinite(context).all():
        tokens[0] = context
        mask[0] = True
    if objects is not None:
        obj = np.asarray(objects, dtype=np.float32)
        if obj.ndim != 2 or obj.shape[1] != 16:
            raise ValueError("objects must have shape [M, 16]")
        obj = obj[np.isfinite(obj).all(axis=1)][:max_objects]
        tokens[1:1 + len(obj)] = obj
        mask[1:1 + len(obj)] = True
    radar_map = radars or {}
    radar_presence = np.zeros(4, dtype=np.bool_)
    offset = 1 + max_objects
    for side_i, side in enumerate(("front", "left", "right", "rear")):
        readings = radar_map.get(side)
        if readings is None:
            continue
        radar_presence[side_i] = (bool(radar_valid[side]) if radar_valid is not None
                                  and side in radar_valid else True)
        if not radar_presence[side_i]:
            continue
        rows = np.asarray(readings, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] > 16:
            raise ValueError(f"radars[{side!r}] must have shape [N, F<=16]")
        rows = rows[np.isfinite(rows).all(axis=1)]
        if rows.shape[1] >= 4:
            rows = rows[np.argsort(np.linalg.norm(rows[:, :3], axis=1), kind="stable")]
            unique: list[np.ndarray] = []
            for row in rows:
                if all(np.linalg.norm(row[:3] - prior[:3]) >= 0.5 for prior in unique):
                    unique.append(row)
            rows = np.asarray(unique, dtype=np.float32).reshape(-1, rows.shape[1])
            rows[:, :3] /= 100.0
            rows[:, 3] /= 30.0
            rows = rows[:radar_capacity]
        start = offset + side_i * radar_capacity
        tokens[start:start + len(rows), :rows.shape[1]] = rows
        mask[start:start + len(rows)] = True
    if rgb is None:
        pixels = np.zeros((3, model_size[1], model_size[0]), dtype=np.uint8)
        pixels_valid = np.asarray(False, dtype=np.bool_)
    else:
        pixels = _resize_rgb(rgb, model_size).transpose(2, 0, 1).copy()
        pixels_valid = np.asarray(True, dtype=np.bool_)
    return {"pixels": pixels, "tokens": tokens, "mask": mask,
            "pixels_valid": pixels_valid, "radar_valid": radar_presence}
