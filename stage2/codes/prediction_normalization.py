#!/usr/bin/env python3
"""Shared parsing and sparse normalization for evaluator predictions.

Both unconstrained and constrained arms pass through these functions before
scoring.  ``Parsed Prediction Dense`` can therefore preserve the exact parsed
model output while ``Prediction`` contains the canonical sparse object used by
the scorer.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from dynamic_template_constraint import canonicalize_legacy_keys, coerce_canonical_types
from nodule_scoring import is_null, norm_value


def parse_json_object(value: Any) -> dict[str, Any] | None:
    """Parse a JSON object, returning ``None`` when no object can be parsed."""
    if isinstance(value, dict):
        return deepcopy(value)
    if not isinstance(value, str):
        return None

    value = value.strip()
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"(\{.*\})", value, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
    return None


def safe_json_loads(value: Any) -> dict[str, Any]:
    """Backward-compatible object parser used by inference."""
    parsed = parse_json_object(value)
    return parsed if parsed is not None else {}


def remove_nulls(value: Any) -> Any:
    """Recursively remove JSON nulls, null sentinels, and empty containers."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            normalized = remove_nulls(item)
            if is_null(normalized):
                continue
            cleaned[key] = normalized
        return cleaned

    if isinstance(value, list):
        cleaned = []
        for item in value:
            normalized = remove_nulls(item)
            if is_null(normalized):
                continue
            cleaned.append(normalized)
        return cleaned

    return None if is_null(value) else value


def normalize_prediction(
    prediction: Any,
    template_style: str = "canonical",
) -> dict[str, Any]:
    """Convert dense/null output to the canonical sparse evaluation form."""
    if not isinstance(prediction, dict):
        return {}

    # Normalization must never modify Parsed Prediction Dense in place.
    prediction = deepcopy(prediction)

    if template_style == "legacy_snake_case":
        prediction = canonicalize_legacy_keys(prediction)

    cleaned = remove_nulls(prediction)
    if not isinstance(cleaned, dict):
        return {}
    cleaned = coerce_canonical_types(cleaned)

    # Dense count-zero output is equivalent to sparse no-nodule output. Keep
    # any real report-level fields, but remove structural count/array holders.
    raw_count = norm_value(prediction.get("Number of Nodules"))
    nodules = cleaned.get("Nodules")
    if raw_count == 0.0 and not nodules:
        cleaned.pop("Number of Nodules", None)
        cleaned.pop("Nodules", None)

    return cleaned
