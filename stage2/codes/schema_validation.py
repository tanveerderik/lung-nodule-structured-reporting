#!/usr/bin/env python3
"""Strict pre-normalization JSON-Schema validation for saved predictions."""

from __future__ import annotations

import re
from typing import Any


def validate_raw_prediction(
    value: Any,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Validate the untouched parsed object; never apply semantic aliases."""
    if not isinstance(value, dict):
        return {
            "valid": False,
            "error_count": 1,
            "errors": [{"path": "$", "message": "parsed prediction is not an object"}],
        }

    details = _validate(value, schema, "$")
    return {"valid": not details, "error_count": len(details), "errors": details}


def _error(path: str, message: str, validator: str) -> dict[str, str]:
    return {"path": path, "message": message, "validator": validator}


def _validate(value: Any, schema: dict[str, Any], path: str) -> list[dict[str, str]]:
    """Validate the JSON-Schema subset used by the supplied nodule schema."""
    if "anyOf" in schema:
        branches = [_validate(value, branch, path) for branch in schema["anyOf"]]
        if any(not errors for errors in branches):
            return []
        # Report the closest branch failure. This preserves a useful nested
        # JSON path (for example $.Nodules[0].Calcification Patterns) instead
        # of collapsing every error to the outer anyOf container.
        def branch_type_matches(branch: dict[str, Any]) -> bool:
            expected = branch.get("type")
            return {
                "null": value is None,
                "object": isinstance(value, dict),
                "array": isinstance(value, list),
                "string": isinstance(value, str),
                "integer": isinstance(value, int) and not isinstance(value, bool),
                "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            }.get(expected, False)

        matching = [
            errors
            for branch, errors in zip(schema["anyOf"], branches)
            if branch_type_matches(branch)
        ]
        candidates = matching or branches
        return min(
            candidates,
            key=lambda errors: (
                len(errors),
                -max((len(error["path"]) for error in errors), default=len(path)),
            ),
        )

    expected = schema.get("type")
    type_valid = {
        "null": value is None,
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
    }.get(expected, True)
    if not type_valid:
        return [
            _error(
                path,
                f"expected {expected}, got {type(value).__name__}",
                "type",
            )
        ]

    errors: list[dict[str, str]] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(_error(path, "value is not in enum", "enum"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(_error(path, "value is below minimum", "minimum"))
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(_error(path, "value is above maximum", "maximum"))
    if isinstance(value, str) and "pattern" in schema:
        if re.fullmatch(schema["pattern"], value) is None:
            errors.append(_error(path, "string does not match pattern", "pattern"))
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(_error(path, "array is shorter than minItems", "minItems"))
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(_error(path, "array is longer than maxItems", "maxItems"))
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(_validate(item, item_schema, f"{path}[{index}]"))
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                errors.append(
                    _error(path, f"missing required property {required!r}", "required")
                )
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(
                        _error(f"{path}.{key}", "additional property", "additionalProperties")
                    )
        for key, item in value.items():
            if key in properties:
                errors.extend(_validate(item, properties[key], f"{path}.{key}"))
    return errors
