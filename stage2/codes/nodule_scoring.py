#!/usr/bin/env python3
"""Shared lightweight nodule-level evaluation logic."""

from __future__ import annotations

import re
from typing import Any

from scipy.optimize import linear_sum_assignment

from date_normalization import canonical_date_key, is_date_field


# The single semantic missing-value policy used by every Base evaluation arm.
# Matching is case-insensitive after surrounding whitespace is removed.
# Deliberately excluded: absent, negative, no, normal, unknown, not seen, and
# not identified.  Those strings may encode meaningful clinical assertions.
SEMANTIC_NULL_ALIASES = frozenset({"", "null", "none", "n/a"})

MATCH_WEIGHTS = {
    "Series ID": 10,
    "Image ID": 10,
    "Lobe": 4,
    "Segment": 2,
    "Type": 2,
    "Long Axis (mm)": 3,
    "Short Axis (mm)": 3,
    "Average Diameter (mm)": 3,
}


def is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, dict)):
        return len(value) == 0
    if isinstance(value, str):
        return value.strip().casefold() in SEMANTIC_NULL_ALIASES
    return False


def norm_value(value: Any, field_name: str | None = None) -> Any:
    if is_null(value):
        return None
    if is_date_field(field_name):
        date_key = canonical_date_key(value)
        if date_key is not None:
            return date_key
    if isinstance(value, str):
        value = value.strip()
        try:
            return float(value)
        except Exception:
            return re.sub(r"\s+", " ", value.lower())
    if isinstance(value, (int, float)):
        return float(value)
    return value


def values_equal(
    left: Any,
    right: Any,
    numeric_tol: float = 0.01,
    field_name: str | None = None,
) -> bool:
    left = norm_value(left, field_name=field_name)
    right = norm_value(right, field_name=field_name)
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if isinstance(left, float) and isinstance(right, float):
        return abs(left - right) <= numeric_tol
    return left == right


def clean_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if not is_null(item)}


def extract_nodules(obj: Any) -> list[dict[str, Any]]:
    if not isinstance(obj, dict):
        return []
    for key in ["nodules", "Nodules", "Nodules ", "Nodule", "Nodules (Array)"]:
        if key in obj and isinstance(obj[key], list):
            return [clean_dict(item) for item in obj[key] if isinstance(item, dict)]
    for value in obj.values():
        if isinstance(value, dict):
            found = extract_nodules(value)
            if found:
                return found
        elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
            fields: set[str] = set()
            for item in value:
                fields.update(item.keys())
            if {"Lobe", "Type", "Image ID", "Series ID", "Long Axis (mm)"} & fields:
                return [clean_dict(item) for item in value]
    return []


def nodule_direct_key(nodule: dict[str, Any]) -> tuple[Any, Any] | None:
    series = norm_value(nodule.get("Series ID"))
    image = norm_value(nodule.get("Image ID"))
    return (series, image) if series is not None and image is not None else None


def nodule_similarity(gold: dict[str, Any], pred: dict[str, Any]) -> float:
    total = matched = 0.0
    for field, weight in MATCH_WEIGHTS.items():
        if field in gold and not is_null(gold[field]):
            total += weight
            if field in pred and values_equal(gold[field], pred[field], field_name=field):
                matched += weight
    return matched / total if total > 0 else 0.0


def match_nodules(
    gold_nodules: list[dict[str, Any]],
    pred_nodules: list[dict[str, Any]],
    min_similarity: float = 0.20,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    matches: list[tuple[int, int, float]] = []
    used_g: set[int] = set()
    used_p: set[int] = set()
    pred_key_map: dict[tuple[Any, Any], list[int]] = {}
    for pred_index, pred in enumerate(pred_nodules):
        key = nodule_direct_key(pred)
        if key is not None:
            pred_key_map.setdefault(key, []).append(pred_index)
    for gold_index, gold in enumerate(gold_nodules):
        key = nodule_direct_key(gold)
        if key is not None and key in pred_key_map:
            for pred_index in pred_key_map[key]:
                if pred_index not in used_p:
                    matches.append((gold_index, pred_index, 1.0))
                    used_g.add(gold_index)
                    used_p.add(pred_index)
                    break

    remaining_g = [i for i in range(len(gold_nodules)) if i not in used_g]
    remaining_p = [i for i in range(len(pred_nodules)) if i not in used_p]
    if remaining_g and remaining_p:
        similarities = [
            [nodule_similarity(gold_nodules[i], pred_nodules[j]) for j in remaining_p]
            for i in remaining_g
        ]
        rows, columns = linear_sum_assignment([[-value for value in row] for row in similarities])
        for row, column in zip(rows, columns):
            gold_index = remaining_g[row]
            pred_index = remaining_p[column]
            similarity = similarities[row][column]
            if similarity >= min_similarity:
                matches.append((gold_index, pred_index, similarity))
                used_g.add(gold_index)
                used_p.add(pred_index)

    return (
        matches,
        set(range(len(gold_nodules))) - used_g,
        set(range(len(pred_nodules))) - used_p,
    )


def score_feature_dict(
    gold: dict[str, Any], pred: dict[str, Any]
) -> tuple[float, float, float]:
    tp = fp = fn = 0.0
    gold = clean_dict(gold)
    pred = clean_dict(pred)
    gold_fields = set(gold)
    pred_fields = set(pred)
    for field in gold_fields:
        if field not in pred_fields:
            fn += 2
        elif values_equal(gold[field], pred[field], field_name=field):
            tp += 2
        else:
            tp += 1
            fp += 0.5
            fn += 0.5
    for field in pred_fields - gold_fields:
        fp += 2
    return tp, fp, fn


def score_case(
    gold_obj: dict[str, Any],
    pred_obj: dict[str, Any],
    min_similarity: float = 0.20,
) -> dict[str, float | int]:
    gold_nodules = extract_nodules(gold_obj)
    pred_nodules = extract_nodules(pred_obj)
    matches, unmatched_g, unmatched_p = match_nodules(
        gold_nodules, pred_nodules, min_similarity=min_similarity
    )
    tp = fp = fn = 0.0
    for gold_index, pred_index, _ in matches:
        match_tp, match_fp, match_fn = score_feature_dict(
            gold_nodules[gold_index], pred_nodules[pred_index]
        )
        tp += match_tp
        fp += match_fp
        fn += match_fn
    for gold_index in unmatched_g:
        fn += 2 * len(clean_dict(gold_nodules[gold_index]))
    for pred_index in unmatched_p:
        fp += 2 * len(clean_dict(pred_nodules[pred_index]))

    if not gold_nodules and not pred_nodules:
        precision = recall = f1 = iou = 1.0
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": iou,
        "Gold Nodule Count": len(gold_nodules),
        "Predicted Nodule Count": len(pred_nodules),
        "Matched Nodule Count": len(matches),
    }
