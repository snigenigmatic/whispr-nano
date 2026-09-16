"""Validated Pseudo2Real parameter arithmetic."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping


def _signature(value: Any) -> tuple[tuple[int, ...], str]:
    return tuple(int(v) for v in value.shape), str(value.dtype)


def validate_compatible(*states: Mapping[str, Any]) -> list[str]:
    if len(states) < 2:
        raise ValueError("at least two states are required")
    keys = sorted(states[0])
    for state in states[1:]:
        if sorted(state) != keys:
            raise ValueError("checkpoint parameter keys differ")
    for key in keys:
        signatures = {_signature(state[key]) for state in states}
        if len(signatures) != 1:
            raise ValueError(f"checkpoint tensor signature differs: {key}")
    return keys


def apply_global_correction(
    source_real: Mapping[str, Any],
    source_pseudo: Mapping[str, Any],
    target_pseudo: Mapping[str, Any],
    scale: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return theta_target + scale * (theta_source_real-theta_source_pseudo)."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    keys = validate_compatible(source_real, source_pseudo, target_pseudo)
    corrected: dict[str, Any] = {}
    correction_sq = target_sq = 0.0
    for key in keys:
        delta = source_real[key] - source_pseudo[key]
        value = target_pseudo[key] + delta * scale
        if hasattr(value, "isfinite"):
            finite = bool(value.isfinite().all())
        else:
            import numpy as np

            finite = bool(np.isfinite(value).all())
        if not finite:
            raise ValueError(f"non-finite corrected tensor: {key}")
        corrected[key] = value
        correction_sq += _sum_squares(delta)
        target_sq += _sum_squares(target_pseudo[key])
    report = {
        "formula": "target_pseudo + scale * (source_real - source_pseudo)",
        "scale": scale,
        "parameter_tensors": len(keys),
        "correction_l2": math.sqrt(correction_sq),
        "target_l2": math.sqrt(target_sq),
        "relative_correction_l2": math.sqrt(correction_sq / target_sq) if target_sq else None,
    }
    return corrected, report


def _sum_squares(value: Any) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().double().square().sum().cpu())
    import numpy as np

    return float(np.square(np.asarray(value, dtype=np.float64)).sum())


def state_fingerprint(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(_signature(value)).encode("utf-8") + b"\0")
        if hasattr(value, "detach"):
            raw = value.detach().cpu().contiguous().numpy().tobytes()
        else:
            raw = value.tobytes()
        digest.update(raw)
    return digest.hexdigest()
