#!/usr/bin/env python3
"""Precision-preserving date normalization for evaluation.

The datasets contain source-faithful dates at day, month, and year precision.
Padding is presentation, not meaning, so ``3/2/2022`` equals ``03/02/2022``.
Missing precision is meaningful, so ``3/2022`` does not equal ``3/1/2022``.
"""

from __future__ import annotations

import calendar
import re
from typing import Any


DATE_FIELD_NAMES = frozenset({"Comparison Date", "Follow-up Date"})

_FULL_DATE_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})$")
_MONTH_YEAR_RE = re.compile(r"^(\d{1,2})/(\d{4})$")
_YEAR_RE = re.compile(r"^(\d{4})$")


def is_date_field(field_name: Any) -> bool:
    """Return whether a scalar or dotted feature path identifies a date."""
    if not isinstance(field_name, str):
        return False
    return field_name.rsplit(".", 1)[-1].strip() in DATE_FIELD_NAMES


def _expand_two_digit_year(year: int) -> int:
    """Use Python/POSIX's conventional 1969--2068 two-digit-year window."""
    return 2000 + year if year <= 68 else 1900 + year


def canonical_date_key(value: Any) -> tuple[str, int, int | None, int | None] | None:
    """Return a semantic date key, or ``None`` for unrecognized/invalid input.

    The first tuple element encodes precision (``day``, ``month``, or ``year``),
    preventing a partially specified source date from being silently expanded.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        text = str(value)
    elif isinstance(value, str):
        text = value.strip()
    else:
        return None

    match = _FULL_DATE_RE.fullmatch(text)
    if match:
        month, day, raw_year = match.groups()
        month_i = int(month)
        day_i = int(day)
        year_i = int(raw_year)
        if len(raw_year) == 2:
            year_i = _expand_two_digit_year(year_i)
        if not 1 <= year_i <= 9999 or not 1 <= month_i <= 12:
            return None
        if not 1 <= day_i <= calendar.monthrange(year_i, month_i)[1]:
            return None
        return ("day", year_i, month_i, day_i)

    match = _MONTH_YEAR_RE.fullmatch(text)
    if match:
        month_i, year_i = map(int, match.groups())
        if 1 <= month_i <= 12 and 1 <= year_i <= 9999:
            return ("month", year_i, month_i, None)
        return None

    match = _YEAR_RE.fullmatch(text)
    if match:
        year_i = int(match.group(1))
        if 1 <= year_i <= 9999:
            return ("year", year_i, None, None)

    return None


DATE_SCORING_POLICY = (
    "calendar-valid date comparison with optional zero-padding; two-digit years "
    "use the 1969-2068 window; day/month/year precision is preserved"
)
