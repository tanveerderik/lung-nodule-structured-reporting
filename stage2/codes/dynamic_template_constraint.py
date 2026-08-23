#!/usr/bin/env python3
"""Corrected dynamic template-constrained decoding for stock vLLM V0.

This module reproduces the mechanism used by the original modified-vLLM lung
nodule decoder without modifying vLLM itself:

* literal template tokens are forced;
* placeholder slots use per-feature candidate sets;
* small candidate sets use the original prefix-filtered sequence logic;
* ``legacy_compat=True`` keeps the original large-list token vocabulary, but
  validates every composed prefix against the slot's integer, decimal, or date
  type instead of permitting an unbounded bag-of-tokens language;
* the first completed slot determines how many nodule dictionaries are placed
  in the subsequently traversed template; and
* candidate selection is argmax after the legacy sampling transforms.

The original >=50-candidate implementation admitted arbitrary repetition of
any token found anywhere in the candidate list. That language has no maximum
length and, in particular, permits continuations such as ``null333...``. This
module deliberately repairs that defect: complete ``null`` values terminate,
and all other flattened slots are finite typed languages.

The logits processor is deliberately stateless with respect to an individual
sequence. It reconstructs the state from ``output_token_ids`` on every call,
which is safe when vLLM shares/clones a processor across child sequences.
"""

from __future__ import annotations

import copy
import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


PROCESSOR_QUALNAME = (
    "dynamic_template_constraint.DynamicTemplateLogitsProcessor"
)
CONTROL_CANDIDATES = {"string", "no comma"}
PLACEHOLDER_RE = re.compile(r"^<\|[a-z0-9_]+\|>$")


CANONICAL_KEY_BY_MARKER = {
    "<|num_nodule|>": "Number of Nodules",
    "<|nodule_id|>": "Nodule ID",
    "<|series|>": "Series ID",
    "<|image|>": "Image ID",
    "<|lobe|>": "Lobe",
    "<|segment|>": "Segment",
    "<|fissure|>": "Fissure",
    "<|pleural|>": "Peripheral",
    "<|tracheobronchial|>": "Tracheobronchial",
    "<|perihilar|>": "Perihilar",
    "<|lung|>": "Lung",
    "<|type|>": "Type",
    "<|calcification_patterns|>": "Calcification Patterns",
    "<|margin|>": "Margin",
    "<|shape|>": "Shape",
    "<|long_axis|>": "Long Axis (mm)",
    "<|short_axis|>": "Short Axis (mm)",
    "<|third_axis|>": "Third Axis (mm)",
    "<|average_diameter|>": "Average Diameter (mm)",
    "<|part_solid_diameter|>": "Part-Solid Diameter (mm)",
    "<|volume|>": "Volume (mm^3)",
    "<|mass|>": "Mass (mg)",
    "<|qualitative_descriptor|>": "Qualitative Size",
    "<|status|>": "Stability",
    "<|lung_rads|>": "Lung-RADS",
    "<|comparison_date|>": "Comparison Date",
    "<|overall_lung_rads|>": "Overall Lung-RADS",
    "<|recommend_imaging|>": "Recommend Imaging",
    "<|recommend_interval|>": "Imaging Interval",
    "<|follow_up_date|>": "Follow-up Date",
}

LEGACY_KEY_BY_MARKER = {
    marker: marker.removeprefix("<|").removesuffix("|>")
    for marker in CANONICAL_KEY_BY_MARKER
}

CANONICAL_TOP_LEVEL_KEYS = (
    "Number of Nodules",
    "Nodules",
    "Overall Lung-RADS",
    "Recommend Imaging",
    "Imaging Interval",
    "Follow-up Date",
)

CANONICAL_NODULE_KEYS = tuple(
    CANONICAL_KEY_BY_MARKER[marker]
    for marker in (
        "<|nodule_id|>",
        "<|series|>",
        "<|image|>",
        "<|lobe|>",
        "<|segment|>",
        "<|fissure|>",
        "<|pleural|>",
        "<|tracheobronchial|>",
        "<|perihilar|>",
        "<|lung|>",
        "<|type|>",
        "<|calcification_patterns|>",
        "<|margin|>",
        "<|shape|>",
        "<|long_axis|>",
        "<|short_axis|>",
        "<|third_axis|>",
        "<|average_diameter|>",
        "<|part_solid_diameter|>",
        "<|volume|>",
        "<|mass|>",
        "<|qualitative_descriptor|>",
        "<|status|>",
        "<|lung_rads|>",
        "<|comparison_date|>",
    )
)

INTEGER_FIELDS = {
    "Number of Nodules",
    "Nodule ID",
    "Series ID",
    "Image ID",
}

FLOAT_FIELDS = {
    "Long Axis (mm)",
    "Short Axis (mm)",
    "Third Axis (mm)",
    "Average Diameter (mm)",
    "Part-Solid Diameter (mm)",
    "Volume (mm^3)",
    "Mass (mg)",
}

LEGACY_TO_CANONICAL_KEYS = {
    "num_nodule": "Number of Nodules",
    "nodules": "Nodules",
    "nodule_id": "Nodule ID",
    "series": "Series ID",
    "image": "Image ID",
    "lobe": "Lobe",
    "segment": "Segment",
    "fissure": "Fissure",
    "pleural": "Peripheral",
    "tracheobronchial": "Tracheobronchial",
    "perihilar": "Perihilar",
    "lung": "Lung",
    "type": "Type",
    "calcification_patterns": "Calcification Patterns",
    "margin": "Margin",
    "shape": "Shape",
    "long_axis": "Long Axis (mm)",
    "short_axis": "Short Axis (mm)",
    "third_axis": "Third Axis (mm)",
    "average_diameter": "Average Diameter (mm)",
    "part_solid_diameter": "Part-Solid Diameter (mm)",
    "volume": "Volume (mm^3)",
    "mass": "Mass (mg)",
    "qualitative_descriptor": "Qualitative Size",
    "status": "Stability",
    "lung_rads": "Lung-RADS",
    "comparison_date": "Comparison Date",
    "overall_lung_rads": "Overall Lung-RADS",
    "recommend_imaging": "Recommend Imaging",
    "recommend_interval": "Imaging Interval",
    "follow_up_date": "Follow-up Date",
}

MARKER_BY_CANONICAL_KEY = {
    canonical: marker for marker, canonical in CANONICAL_KEY_BY_MARKER.items()
}

_INTEGER_MAX_BY_MARKER = {
    "<|num_nodule|>": 49,
}

_COMPOSED_INTEGER_MARKERS = frozenset({
    "<|nodule_id|>",
    "<|series|>",
    "<|image|>",
})

_COMPOSED_FLOAT_MARKERS = frozenset({
    "<|long_axis|>",
    "<|short_axis|>",
    "<|third_axis|>",
    "<|average_diameter|>",
    "<|part_solid_diameter|>",
    "<|volume|>",
    "<|mass|>",
})

# These are syntactic safety bounds, not clinical/candidate range claims.
# They keep every composed language finite while remaining well above values
# seen in the supplied reports.
_MAX_COMPOSED_INTEGER_DIGITS = 8
_MAX_COMPOSED_DECIMAL_PLACES = 4

_MONTH_TEXTS = frozenset(
    {str(value) for value in range(1, 13)}
    | {f"{value:02d}" for value in range(1, 13)}
)
_DAY_TEXTS = frozenset(
    {str(value) for value in range(1, 32)}
    | {f"{value:02d}" for value in range(1, 32)}
)
def _prefix_of_any(text: str, values: Iterable[str]) -> bool:
    return any(value.startswith(text) for value in values)


def _date_status(
    text: str,
    *,
    allow_month_year: bool,
) -> tuple[bool, bool]:
    """Match the XGrammar/test-label month-first date language."""

    if text and any(char not in "0123456789/" for char in text):
        return False, False
    parts = text.split("/")
    if len(parts) > 3:
        return False, False

    month = parts[0]
    if len(parts) == 1:
        return _prefix_of_any(month, _MONTH_TEXTS), False

    second = parts[1]
    month_complete = month in _MONTH_TEXTS
    if len(parts) == 2:
        month_day_prefix = (
            month_complete and _prefix_of_any(second, _DAY_TEXTS)
        )
        month_year_prefix = (
            allow_month_year
            and month_complete
            and (second == "" or second.isdigit())
            and len(second) <= 4
        )
        complete_month_year = (
            month_year_prefix and second.isdigit() and len(second) == 4
        )
        return (
            month_day_prefix or month_year_prefix,
            complete_month_year,
        )

    year = parts[2]
    valid_full_date = (
        month_complete
        and second in _DAY_TEXTS
        and (year == "" or year.isdigit())
        and len(year) <= 4
    )
    complete_full_date = (
        valid_full_date and year.isdigit() and len(year) in {2, 4}
    )
    return (
        valid_full_date,
        complete_full_date,
    )


def _integer_status(text: str, maximum: int) -> tuple[bool, bool]:
    if not text or not text.isdigit():
        return False, False
    if len(text) > 1 and text.startswith("0"):
        return False, False
    if len(text) > len(str(maximum)):
        return False, False
    value = int(text)
    valid = value <= maximum
    return valid, valid


def _composed_integer_status(text: str) -> tuple[bool, bool]:
    if not text or not text.isdigit():
        return False, False
    if len(text) > 1 and text.startswith("0"):
        return False, False
    valid = len(text) <= _MAX_COMPOSED_INTEGER_DIGITS
    return valid, valid


def _float_status(text: str) -> tuple[bool, bool]:
    """Finite non-negative decimal language with up to four decimal places."""

    if not text or text.count(".") > 1:
        return False, False
    integer, dot, fraction = text.partition(".")
    if not integer.isdigit() or (len(integer) > 1 and integer.startswith("0")):
        return False, False
    if dot and (not fraction.isdigit() and fraction != ""):
        return False, False
    if len(integer) > _MAX_COMPOSED_INTEGER_DIGITS:
        return False, False
    if len(fraction) > _MAX_COMPOSED_DECIMAL_PLACES:
        return False, False
    complete = not dot or bool(fraction)
    return True, complete


def _flat_slot_status(marker: str, value_text: str) -> tuple[bool, bool]:
    """Return whether a large-list value is a valid prefix and/or complete."""

    text = value_text.strip()
    allow_null = marker != "<|num_nodule|>"
    null_prefix = allow_null and "null".startswith(text)
    null_complete = allow_null and text == "null"

    if marker in _INTEGER_MAX_BY_MARKER:
        valid, complete = _integer_status(text, _INTEGER_MAX_BY_MARKER[marker])
    elif marker in _COMPOSED_INTEGER_MARKERS:
        valid, complete = _composed_integer_status(text)
    elif marker in _COMPOSED_FLOAT_MARKERS:
        valid, complete = _float_status(text)
    elif marker in {"<|comparison_date|>", "<|follow_up_date|>"}:
        valid, complete = _date_status(
            text,
            allow_month_year=marker == "<|follow_up_date|>",
        )
    else:
        raise ValueError(f"No typed flattened grammar for marker {marker!r}")

    return valid or null_prefix, complete or null_complete


def legacy_candidate_values() -> dict[str, list[Any]]:
    """Return the candidate lists exactly as defined in the old paper code."""

    return {
        "<|num_nodule|>": list(range(50)),
        "<|nodule_id|>": list(range(50)) + ["null"],
        "<|series|>": list(range(1000)) + ["null"],
        "<|image|>": list(range(2000)) + ["null"],
        "<|lobe|>": [
            "right upper lobe", "right middle lobe", "right lower lobe",
            "left upper lobe", "lingula", "left lower lobe", "null",
            "string",
        ],
        "<|segment|>": [
            "apical", "posterior", "anterior", "lateral", "medial",
            "superior", "medial basal", "anterior basal", "lateral basal",
            "posterior basal", "apico-posterior", "anteromedial basal",
            "inferior", "null", "string",
        ],
        "<|fissure|>": [
            "right minor fissure", "right major fissure", "left major fissure",
            "right fissural", "left fissural", "fissural",
            "right perifissural", "left perifissural", "perifissural",
            "null", "string",
        ],
        "<|pleural|>": [
            "subpleural", "pleural-based", "peripheral", "null", "string",
        ],
        "<|tracheobronchial|>": [
            "airway", "tracheal", "endotracheal", "endobronchial",
            "peribronchial", "peribronchovascular", "bronchovascular",
            "bronchocentric", "null", "string",
        ],
        "<|perihilar|>": ["perihilar", "null", "string"],
        "<|lung|>": ["left", "right", "bilateral", "null", "string"],
        "<|type|>": [
            "solid", "part-solid", "mixed attenuation", "cystic",
            "ground glass", "hazy", "nonsolid", "fluid/water", "fat",
            "calcified", "part-calcified", "noncalcified", "cavitary",
            "null", "string",
        ],
        "<|calcification_patterns|>": [
            "diffuse", "central", "lamellated", "popcorn", "eccentric",
            "dense", "dendriform", "punctate", "linear", "null", "string",
        ],
        "<|margin|>": [
            "spiculated", "smooth", "lobulated", "fuzzy", "irregular",
            "null", "string",
        ],
        "<|shape|>": [
            "oval", "lentiform", "triangular", "round", "bilobed",
            "rectangular", "polygonal", "spherical", "irregular", "ovoid",
            "tubular", "branching", "null", "string",
        ],
        "<|long_axis|>": list(range(101)) + ["0.0", "null"],
        "<|short_axis|>": list(range(101)) + ["0.0", "null"],
        "<|third_axis|>": list(range(101)) + ["0.0", "null"],
        "<|average_diameter|>": list(range(101)) + ["0.0", "null"],
        "<|part_solid_diameter|>": list(range(101)) + ["0.0", "null"],
        "<|volume|>": list(range(10000)) + ["0.0", "null"],
        "<|mass|>": list(range(10000)) + ["0.0", "null"],
        "<|qualitative_descriptor|>": [
            "large", "small", "tiny", "micronodule", "punctate",
            "scattered micronodules", "subcentimeter", "null", "string",
        ],
        "<|status|>": [
            "stable", "increase", "decrease", "new", "changed", "baseline",
            "interval development", "resolved", "null", "string",
        ],
        "<|lung_rads|>": [
            "0", "1", "2", "3", "4A", "4B", "4X", "null", "string",
        ],
        "<|comparison_date|>": (
            [f"{i:02}/01/2024" for i in range(1, 13)]
            + [f"01/{i:02}/2024" for i in range(1, 32)]
            + [f"01/01/{i:04}" for i in range(2000, 2026)]
            + ["null", "string"]
        ),
        "<|overall_lung_rads|>": [
            "0", "1", "2", "3", "4A", "4B", "4X", "null", "string",
            "no comma",
        ],
        "<|recommend_imaging|>": [
            "LDCT",
            "LDCT, PET/CT",
            "LDCT, Tissue sampling",
            "LDCT, PET/CT, Tissue sampling",
            "LDCT, Diagnostic CT",
            "LDCT, Diagnostic CT, PET/CT",
            "LDCT, Diagnostic CT, PET/CT, Tissue sampling",
            "Diagnostic CT",
            "Diagnostic CT with contrast",
            "Diagnostic CT, PET/CT",
            "Diagnostic CT with contrast, PET/CT",
            "Diagnostic CT, PET/CT, Tissue sampling",
            "Diagnostic CT with contrast, PET/CT, Tissue sampling",
            "PET/CT",
            "PET/CT, Tissue sampling",
            "Tissue sampling",
            "null",
            "string",
        ],
        "<|recommend_interval|>": [
            "1 month", "1-2 months", "1-3 months", "2-3 months",
            "3 months", "3-6 months", "4-6 months", "6 months",
            "6-12 months", "12 months", "null", "string",
        ],
        "<|follow_up_date|>": (
            [f"{i:02}/01/2024" for i in range(1, 13)]
            + [f"01/{i:02}/2024" for i in range(1, 32)]
            + [f"01/01/{i:04}" for i in range(2000, 2026)]
            + ["null", "string", "no comma"]
        ),
    }


def _nodule_template(key_by_marker: Mapping[str, str]) -> dict[str, str]:
    markers = (
        "<|nodule_id|>", "<|series|>", "<|image|>", "<|lobe|>",
        "<|segment|>", "<|fissure|>", "<|pleural|>",
        "<|tracheobronchial|>", "<|perihilar|>", "<|lung|>",
        "<|type|>", "<|calcification_patterns|>", "<|margin|>",
        "<|shape|>", "<|long_axis|>", "<|short_axis|>",
        "<|third_axis|>", "<|average_diameter|>",
        "<|part_solid_diameter|>", "<|volume|>", "<|mass|>",
        "<|qualitative_descriptor|>", "<|status|>", "<|lung_rads|>",
        "<|comparison_date|>",
    )
    return {key_by_marker[marker]: marker for marker in markers}


def build_dense_template(
    nodule_count: int,
    template_style: str = "canonical",
) -> dict[str, Any]:
    """Build the ordered dense template after count-dependent expansion."""

    if nodule_count < 0:
        raise ValueError("nodule_count must be non-negative")
    if template_style == "canonical":
        keys = CANONICAL_KEY_BY_MARKER
        nodules_key = "Nodules"
    elif template_style == "legacy_snake_case":
        keys = LEGACY_KEY_BY_MARKER
        nodules_key = "nodules"
    else:
        raise ValueError(
            "template_style must be 'canonical' or 'legacy_snake_case'"
        )

    template: dict[str, Any] = {
        keys["<|num_nodule|>"]: "<|num_nodule|>",
    }
    if nodule_count > 0:
        template[nodules_key] = [
            copy.deepcopy(_nodule_template(keys))
            for _ in range(nodule_count)
        ]
    template[keys["<|overall_lung_rads|>"]] = "<|overall_lung_rads|>"
    template[keys["<|recommend_imaging|>"]] = "<|recommend_imaging|>"
    template[keys["<|recommend_interval|>"]] = "<|recommend_interval|>"
    template[keys["<|follow_up_date|>"]] = "<|follow_up_date|>"
    return template


@dataclass(frozen=True)
class LiteralNode:
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class CandidateEntry:
    value_text: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class CandidateSet:
    marker: str
    label: str
    source_value_count: int
    entries: tuple[CandidateEntry, ...]
    flattened_token_ids: tuple[int, ...]
    invisible_leading_prefixes: tuple[tuple[int, ...], ...]
    legacy_flattened: bool

    @property
    def mode(self) -> str:
        return "typed_flattened" if self.legacy_flattened else "trie"


@dataclass(frozen=True)
class SlotNode:
    marker: str
    label: str
    candidates: CandidateSet


ProgramNode = LiteralNode | SlotNode


@dataclass(frozen=True)
class CompiledProgram:
    nodes: tuple[ProgramNode, ...]
    nodule_count: int
    template_text: str


@dataclass(frozen=True)
class NextAction:
    kind: str
    token_id: Optional[int] = None
    allowed_token_ids: tuple[int, ...] = ()
    restrict_to_allowed: bool = True
    slot_label: Optional[str] = None
    slot_marker: Optional[str] = None
    slot_mode: Optional[str] = None
    slot_prefix: tuple[int, ...] = ()


@dataclass(frozen=True)
class SlotTrace:
    slot: str
    marker: str
    mode: str
    chosen: str
    candidate_count: int
    allowed_values: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        out = {
            "slot": self.slot,
            "marker": self.marker,
            "mode": self.mode,
            "chosen": self.chosen,
            "candidate_count": self.candidate_count,
        }
        if self.allowed_values:
            out["allowed_values"] = list(self.allowed_values)
        return out


@dataclass(frozen=True)
class ReplayResult:
    action: NextAction
    traces: tuple[SlotTrace, ...]
    count_value: Optional[int]
    program_complete: bool


class TemplateStateError(RuntimeError):
    pass


class DynamicTemplateRuntime:
    """Tokenizer-compiled immutable template and candidate data."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        template_style: str = "canonical",
        include_json_tags: bool = True,
        json_begin_tag: str = "<json>",
        json_end_tag: str = "</json>",
        json_indent: int = 4,
        legacy_compat: bool = True,
        template_add_special_tokens: bool = False,
        max_dynamic_nodules: Optional[int] = 49,
        original_vocab_size: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.template_style = template_style
        self.include_json_tags = include_json_tags
        self.json_begin_tag = json_begin_tag
        self.json_end_tag = json_end_tag
        self.json_indent = json_indent
        self.legacy_compat = legacy_compat
        self.template_add_special_tokens = template_add_special_tokens
        self.max_dynamic_nodules = max_dynamic_nodules
        self.original_vocab_size = original_vocab_size

        candidate_values = legacy_candidate_values()
        self.markers = tuple(candidate_values)
        self.marker_ids = self._register_and_audit_markers(self.markers)
        self.marker_by_id = {
            token_id: marker for marker, token_id in self.marker_ids.items()
        }
        self.candidate_sets = self._compile_candidate_sets(candidate_values)
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if self.eos_token_id is None:
            raise ValueError("Tokenizer has no eos_token_id")

        self._programs: dict[int, CompiledProgram] = {}
        self._program_lock = threading.RLock()
        self.seed_program = self.get_program(1)

    @classmethod
    def from_pretrained(cls, tokenizer_dir: str, **kwargs: Any):
        from transformers import AutoConfig, AutoTokenizer

        markers = list(legacy_candidate_values())
        base_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            trust_remote_code=True,
        )
        model_config = AutoConfig.from_pretrained(
            tokenizer_dir,
            trust_remote_code=True,
        )
        original_vocab_size = int(
            getattr(model_config, "vocab_size", len(base_tokenizer))
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            trust_remote_code=True,
            additional_special_tokens=markers,
        )
        return cls(
            tokenizer,
            original_vocab_size=original_vocab_size,
            **kwargs,
        )

    def _register_and_audit_markers(
        self,
        markers: Sequence[str],
    ) -> dict[str, int]:
        additional = list(getattr(self.tokenizer, "additional_special_tokens", []))
        missing = [marker for marker in markers if marker not in additional]
        if missing and hasattr(self.tokenizer, "add_special_tokens"):
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": list(missing)}
            )

        marker_ids: dict[str, int] = {}
        for marker in markers:
            ids = self.encode(marker, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(
                    f"Placeholder {marker!r} must encode to one private special "
                    f"token, got {ids!r}"
                )
            marker_ids[marker] = ids[0]
        if len(set(marker_ids.values())) != len(marker_ids):
            raise ValueError("Placeholder markers do not have unique token IDs")
        return marker_ids

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return [int(x) for x in ids]

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def decode_output(self, token_ids: Sequence[int]) -> str:
        """Decode as vLLM's default text output (special tokens omitted)."""
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def compiled_program_ends_with_eos(self, count: int) -> bool:
        program = self.get_program(count)
        last = program.nodes[-1]
        return bool(
            isinstance(last, LiteralNode)
            and last.token_ids
            and last.token_ids[-1] == self.eos_token_id
        )

    def _compile_candidate_sets(
        self,
        candidate_values: Mapping[str, Sequence[Any]],
    ) -> dict[str, CandidateSet]:
        compiled: dict[str, CandidateSet] = {}
        for marker, source_values in candidate_values.items():
            entries: list[CandidateEntry] = []
            flat: list[int] = []
            source_count = len(source_values)
            for value in source_values:
                if value in CONTROL_CANDIDATES:
                    continue
                text = str(value)
                token_ids = tuple(self.encode(text, add_special_tokens=False))
                if not token_ids:
                    raise ValueError(f"Candidate {marker}={text!r} encoded empty")
                entries.append(CandidateEntry(text, token_ids))
                flat.extend(token_ids)

            use_flat = self.legacy_compat and source_count >= 50
            # Preserve first-seen order for deterministic audits and tests.
            flat_unique = tuple(dict.fromkeys(flat)) if use_flat else ()
            invisible_prefixes: list[tuple[int, ...]] = []
            if use_flat:
                # Some SentencePiece/Tekken tokenizers prepend a boundary token
                # when a standalone scalar is encoded.  Decoding only that
                # token produces an empty string, although decoding the full
                # candidate produces the intended number/date.  Admit only
                # finite leading token prefixes observed in real candidate
                # encodings; arbitrary or repeated zero-progress tokens remain
                # illegal and therefore cannot recreate the old runaway loop.
                for entry in entries:
                    for length in range(1, len(entry.token_ids) + 1):
                        prefix = entry.token_ids[:length]
                        if self.decode(prefix).strip():
                            break
                        if prefix not in invisible_prefixes:
                            invisible_prefixes.append(prefix)
            compiled[marker] = CandidateSet(
                marker=marker,
                label=CANONICAL_KEY_BY_MARKER[marker],
                source_value_count=source_count,
                entries=tuple(entries),
                flattened_token_ids=flat_unique,
                invisible_leading_prefixes=tuple(invisible_prefixes),
                legacy_flattened=use_flat,
            )
        return compiled

    def _render_template_text(self, count: int) -> str:
        body = json.dumps(
            build_dense_template(count, self.template_style),
            indent=self.json_indent,
        )
        if self.include_json_tags:
            return f"{self.json_begin_tag}\n{body}\n{self.json_end_tag}"
        return body

    def _slot_label(self, marker: str, occurrence: int) -> str:
        base = CANONICAL_KEY_BY_MARKER[marker]
        if marker in {"<|num_nodule|>", "<|overall_lung_rads|>",
                      "<|recommend_imaging|>", "<|recommend_interval|>",
                      "<|follow_up_date|>"}:
            return base
        return f"Nodule[{occurrence}].{base}"

    def _compile_program(self, count: int) -> CompiledProgram:
        text = self._render_template_text(count)
        token_ids = self.encode(
            text,
            add_special_tokens=self.template_add_special_tokens,
        )
        nodes: list[ProgramNode] = []
        literal: list[int] = []
        occurrence_by_marker: dict[str, int] = {}

        for token_id in token_ids:
            marker = self.marker_by_id.get(token_id)
            if marker is None:
                literal.append(token_id)
                continue
            if literal:
                nodes.append(LiteralNode(tuple(literal)))
                literal = []
            occurrence = occurrence_by_marker.get(marker, 0)
            occurrence_by_marker[marker] = occurrence + 1
            nodes.append(
                SlotNode(
                    marker=marker,
                    label=self._slot_label(marker, occurrence),
                    candidates=self.candidate_sets[marker],
                )
            )
        if literal:
            nodes.append(LiteralNode(tuple(literal)))

        if not nodes or not isinstance(nodes[0], LiteralNode):
            raise ValueError("Compiled template must start with a literal")
        for i, node in enumerate(nodes):
            if isinstance(node, SlotNode):
                if i + 1 >= len(nodes) or not isinstance(nodes[i + 1], LiteralNode):
                    raise ValueError(f"Slot {node.marker} has no literal terminator")
                if not nodes[i + 1].token_ids:
                    raise ValueError(f"Slot {node.marker} has an empty terminator")

        return CompiledProgram(tuple(nodes), count, text)

    def get_program(self, count: int) -> CompiledProgram:
        if count < 0:
            raise TemplateStateError(f"Negative nodule count: {count}")
        if self.max_dynamic_nodules is not None and count > self.max_dynamic_nodules:
            raise TemplateStateError(
                f"Predicted nodule count {count} exceeds the explicit dynamic "
                f"template cap {self.max_dynamic_nodules}. Set the cap to -1/None "
                "only for exact legacy behavior outside the intended 0..49 domain."
            )
        with self._program_lock:
            program = self._programs.get(count)
            if program is None:
                program = self._compile_program(count)
                self._programs[count] = program
            return program

    def audit(self) -> dict[str, Any]:
        marker_rows = []
        for marker in self.markers:
            candidates = self.candidate_sets[marker]
            token_ids: set[int] = set()
            for entry in candidates.entries:
                token_ids.update(entry.token_ids)
            marker_rows.append({
                "marker": marker,
                "marker_token_id": self.marker_ids[marker],
                "canonical_key": CANONICAL_KEY_BY_MARKER[marker],
                "source_value_count": candidates.source_value_count,
                "effective_candidate_count": len(candidates.entries),
                "mode": candidates.mode,
                "unique_candidate_token_ids": len(token_ids),
                "flattened_token_ids": len(candidates.flattened_token_ids),
                "invisible_leading_prefixes": len(
                    candidates.invisible_leading_prefixes
                ),
            })
        output_token_ids = set()
        for program_count in (0, 1, 2):
            for node in self.get_program(program_count).nodes:
                if isinstance(node, LiteralNode):
                    output_token_ids.update(node.token_ids)
        for candidates in self.candidate_sets.values():
            for entry in candidates.entries:
                output_token_ids.update(entry.token_ids)
        output_token_ids.add(int(self.eos_token_id))
        return {
            "template_style": self.template_style,
            "include_json_tags": self.include_json_tags,
            "json_begin_tag": self.json_begin_tag if self.include_json_tags else None,
            "json_end_tag": self.json_end_tag if self.include_json_tags else None,
            "json_indent": self.json_indent,
            "legacy_compat": self.legacy_compat,
            "template_add_special_tokens": self.template_add_special_tokens,
            "max_dynamic_nodules": self.max_dynamic_nodules,
            "original_vocab_size": self.original_vocab_size,
            "eos_token_id": int(self.eos_token_id),
            "max_output_token_id": max(output_token_ids),
            "output_ids_within_original_vocab": (
                self.original_vocab_size is None
                or max(output_token_ids) < self.original_vocab_size
            ),
            "placeholders": marker_rows,
        }

    def render_tokens(
        self,
        nodule_count: int,
        selections: Optional[Mapping[str, Any]] = None,
    ) -> list[int]:
        """Render one legal token path for unit tests and offline audits.

        ``selections`` may address a slot by its expanded label (for example
        ``Nodule[1].Lobe``), by placeholder marker, or by canonical key. Any
        unspecified non-count slot selects ``null``.
        """

        selections = dict(selections or {})
        program = self.get_program(nodule_count)
        output: list[int] = []
        skip_literal_first = False
        terminators_by_marker: dict[str, list[int]] = {}
        for index, node in enumerate(program.nodes):
            if isinstance(node, LiteralNode):
                start = 1 if skip_literal_first else 0
                output.extend(node.token_ids[start:])
                skip_literal_first = False
                continue

            if node.marker == "<|num_nodule|>":
                value = str(nodule_count)
            else:
                value = selections.get(
                    node.label,
                    selections.get(
                        node.marker,
                        selections.get(CANONICAL_KEY_BY_MARKER[node.marker], "null"),
                    ),
                )
                value = str(value)
            value_ids = tuple(self.encode(value, add_special_tokens=False))
            if node.candidates.legacy_flattened:
                illegal = [
                    token_id for token_id in value_ids
                    if token_id not in node.candidates.flattened_token_ids
                ]
                if illegal:
                    raise ValueError(
                        f"Selection {node.label}={value!r} contains token IDs "
                        f"outside the legacy flattened set: {illegal}"
                    )
                valid, complete = _flat_slot_status(node.marker, value)
                if not valid or not complete:
                    raise ValueError(
                        f"Selection {node.label}={value!r} is not a complete "
                        "value in the corrected typed flattened grammar"
                    )
            elif value_ids not in {
                entry.token_ids for entry in node.candidates.entries
            }:
                raise ValueError(
                    f"Selection {node.label}={value!r} is not a candidate"
                )
            terminator = self._program_terminator(program, index)
            if node.candidates.legacy_flattened:
                output.extend(value_ids)
                output.append(terminator)
            else:
                work = list(value_ids)
                for prior in terminators_by_marker.get(node.marker, ()):
                    if prior not in work:
                        work.append(prior)
                if terminator not in work:
                    work.append(terminator)
                # The old sampler declares completion at the first occurrence
                # of the current terminator, even if it occurs inside a value.
                output.extend(work[:work.index(terminator) + 1])
            seen = terminators_by_marker.setdefault(node.marker, [])
            if terminator not in seen:
                seen.append(terminator)
            skip_literal_first = True
        return output

    @staticmethod
    def _program_terminator(program: CompiledProgram, node_index: int) -> int:
        next_node = program.nodes[node_index + 1]
        if not isinstance(next_node, LiteralNode) or not next_node.token_ids:
            raise ValueError("Slot has no literal terminator")
        return next_node.token_ids[0]


class TemplateStateMachine:
    """Replay generated token IDs and determine the next forced/allowed token."""

    def __init__(self, runtime: DynamicTemplateRuntime) -> None:
        self.runtime = runtime

    @staticmethod
    def _terminator(program: CompiledProgram, node_index: int) -> int:
        next_node = program.nodes[node_index + 1]
        assert isinstance(next_node, LiteralNode)
        return next_node.token_ids[0]

    def _slot_progress(
        self,
        slot: SlotNode,
        terminator: int,
        remaining: Sequence[int],
        prior_terminators: Sequence[int] = (),
    ) -> tuple[int, bool, NextAction, str]:
        prefix: list[int] = []
        candidates = slot.candidates

        if candidates.legacy_flattened:
            for offset, token_id in enumerate(remaining):
                current_text = self.runtime.decode(prefix)
                _, current_complete = _flat_slot_status(
                    slot.marker, current_text
                )
                if token_id == terminator:
                    if not current_complete:
                        raise TemplateStateError(
                            f"Premature terminator in typed flattened slot "
                            f"{slot.label}: {current_text!r}"
                        )
                    return (
                        offset + 1,
                        True,
                        NextAction("complete"),
                        current_text,
                    )
                if token_id not in candidates.flattened_token_ids:
                    raise TemplateStateError(
                        f"Illegal token {token_id} in typed flattened slot "
                        f"{slot.label}"
                    )
                proposal = [*prefix, token_id]
                proposal_text = self.runtime.decode(proposal)
                proposal_valid, _ = _flat_slot_status(
                    slot.marker, proposal_text
                )
                invisible_leading_prefix = (
                    tuple(proposal) in candidates.invisible_leading_prefixes
                )
                if not invisible_leading_prefix and (
                    not proposal_valid
                    or proposal_text.strip() == current_text.strip()
                ):
                    raise TemplateStateError(
                        f"Invalid typed prefix {proposal_text!r} in slot "
                        f"{slot.label}"
                    )
                prefix = proposal

            current_text = self.runtime.decode(prefix)
            _, current_complete = _flat_slot_status(slot.marker, current_text)
            allowed: list[int] = []
            for token_id in candidates.flattened_token_ids:
                if token_id == terminator:
                    continue
                proposal_text = self.runtime.decode([*prefix, token_id])
                proposal_valid, _ = _flat_slot_status(
                    slot.marker, proposal_text
                )
                proposal_ids = (*prefix, token_id)
                invisible_leading_prefix = (
                    proposal_ids in candidates.invisible_leading_prefixes
                )
                if invisible_leading_prefix or (
                    proposal_valid
                    and proposal_text.strip() != current_text.strip()
                ):
                    allowed.append(token_id)
            if current_complete:
                allowed.append(terminator)
            allowed = list(dict.fromkeys(allowed))
            if not allowed:
                raise TemplateStateError(
                    f"No valid continuation for typed flattened slot "
                    f"{slot.label}: {current_text!r}"
                )
            if allowed == [terminator]:
                return (
                    len(remaining),
                    False,
                    NextAction(kind="fixed", token_id=terminator),
                    "",
                )
            return (
                len(remaining),
                False,
                NextAction(
                    kind="candidate",
                    allowed_token_ids=tuple(allowed),
                    restrict_to_allowed=True,
                    slot_label=slot.label,
                    slot_marker=slot.marker,
                    slot_mode=candidates.mode,
                    slot_prefix=tuple(prefix),
                ),
                "",
            )

        survivors: list[tuple[CandidateEntry, list[int]]] = []
        for entry in candidates.entries:
            work = list(entry.token_ids)
            for prior in prior_terminators:
                if prior not in work:
                    work.append(prior)
            # Exact legacy condition: append only if the terminator token does
            # not occur anywhere in the candidate token sequence.
            if terminator not in work:
                work.append(terminator)
            survivors.append((entry, work))

        location = 0
        for offset, token_id in enumerate(remaining):
            allowed = list(set(
                work[location]
                for _, work in survivors
                if location < len(work)
            ))
            if token_id not in allowed:
                raise TemplateStateError(
                    f"Illegal token {token_id} in trie slot {slot.label}; "
                    f"allowed={allowed}, prefix={prefix}"
                )
            survivors = [
                pair for pair in survivors
                if location < len(pair[1]) and pair[1][location] == token_id
            ]
            if token_id == terminator:
                chosen = self.runtime.decode(prefix)
                exact = [
                    entry.value_text
                    for entry, _ in survivors
                    if tuple(prefix) == entry.token_ids
                ]
                if exact:
                    chosen = exact[0]
                return offset + 1, True, NextAction("complete"), chosen
            prefix.append(token_id)
            location += 1

        allowed = list(set(
            work[location]
            for _, work in survivors
            if location < len(work)
        ))
        if not allowed:
            raise TemplateStateError(
                f"No continuation remains in trie slot {slot.label}"
            )
        return (
            len(remaining),
            False,
            NextAction(
                kind="candidate",
                allowed_token_ids=tuple(allowed),
                restrict_to_allowed=True,
                slot_label=slot.label,
                slot_marker=slot.marker,
                slot_mode=candidates.mode,
                slot_prefix=tuple(prefix),
            ),
            "",
        )

    def _replay_program(
        self,
        program: CompiledProgram,
        output_token_ids: Sequence[int],
    ) -> ReplayResult:
        output = [int(x) for x in output_token_ids]
        output_index = 0
        skip_literal_first = False
        traces: list[SlotTrace] = []
        count_value: Optional[int] = None
        terminators_by_marker: dict[str, list[int]] = {}

        for node_index, node in enumerate(program.nodes):
            if isinstance(node, LiteralNode):
                start = 1 if skip_literal_first else 0
                skip_literal_first = False
                for expected in node.token_ids[start:]:
                    if output_index == len(output):
                        return ReplayResult(
                            NextAction(kind="fixed", token_id=expected),
                            tuple(traces), count_value, False,
                        )
                    actual = output[output_index]
                    if actual != expected:
                        raise TemplateStateError(
                            f"Literal divergence at output index {output_index}: "
                            f"expected token {expected}, got {actual}"
                        )
                    output_index += 1
                continue

            terminator = self._terminator(program, node_index)
            consumed, complete, action, chosen = self._slot_progress(
                node,
                terminator,
                output[output_index:],
                terminators_by_marker.get(node.marker, ()),
            )
            output_index += consumed
            if not complete:
                return ReplayResult(
                    action, tuple(traces), count_value, False,
                )

            allowed_values: tuple[str, ...] = ()
            if not node.candidates.legacy_flattened and len(node.candidates.entries) <= 20:
                allowed_values = tuple(
                    entry.value_text for entry in node.candidates.entries
                )
            traces.append(SlotTrace(
                slot=node.label,
                marker=node.marker,
                mode=node.candidates.mode,
                chosen=chosen,
                candidate_count=len(node.candidates.entries),
                allowed_values=allowed_values,
            ))
            if node.marker == "<|num_nodule|>":
                try:
                    count_value = int(chosen.strip())
                except Exception as exc:
                    raise TemplateStateError(
                        f"Legacy count token sequence decoded to {chosen!r}, "
                        "which the original engine could not parse as int"
                    ) from exc
            seen = terminators_by_marker.setdefault(node.marker, [])
            if terminator not in seen:
                seen.append(terminator)
            skip_literal_first = True

        if output_index != len(output):
            trailing = output[output_index:]
            # vLLM V0 may schedule one decode step ahead while asynchronous
            # output processing is enabled.  On that in-flight step the
            # processor is called once more with the EOS token that we forced
            # after completing the template already present in
            # ``output_token_ids``.  Treat exactly that one terminal EOS as an
            # idempotent completed state.  Every non-EOS token (and more than
            # one trailing token) remains a hard invariant violation.
            if trailing != [int(self.runtime.eos_token_id)]:
                raise TemplateStateError(
                    f"Output contains {len(trailing)} token(s) after the "
                    "compiled template completed: {trailing}"
                )
        return ReplayResult(
            NextAction(kind="eos", token_id=int(self.runtime.eos_token_id)),
            tuple(traces), count_value, True,
        )

    def _extract_count(self, output_token_ids: Sequence[int]) -> Optional[int]:
        # Only replay through the first (count) slot. Once count is complete,
        # the correct expanded template can differ immediately after its
        # terminator (most visibly for count zero), so replaying the rest of
        # the one-nodule seed program would reject otherwise valid output.
        program = self.runtime.seed_program
        output = [int(x) for x in output_token_ids]
        output_index = 0
        for node_index, node in enumerate(program.nodes):
            if isinstance(node, LiteralNode):
                for expected in node.token_ids:
                    if output_index == len(output):
                        return None
                    if output[output_index] != expected:
                        raise TemplateStateError(
                            "Literal divergence while locating the count slot: "
                            f"expected {expected}, got {output[output_index]}"
                        )
                    output_index += 1
                continue

            if node.marker != "<|num_nodule|>":
                raise TemplateStateError("Number of Nodules is not the first slot")
            terminator = self._terminator(program, node_index)
            _, complete, _, chosen = self._slot_progress(
                node,
                terminator,
                output[output_index:],
            )
            if not complete:
                return None
            try:
                return int(chosen.strip())
            except Exception as exc:
                raise TemplateStateError(
                    f"Legacy count token sequence decoded to {chosen!r}, "
                    "which the original engine could not parse as int"
                ) from exc
        raise TemplateStateError("Compiled seed template has no count slot")

    def replay(self, output_token_ids: Sequence[int]) -> ReplayResult:
        count = self._extract_count(output_token_ids)
        if count is None:
            return self._replay_program(
                self.runtime.seed_program,
                output_token_ids,
            )
        program = self.runtime.get_program(count)
        replay = self._replay_program(program, output_token_ids)
        if replay.count_value != count:
            raise TemplateStateError(
                f"Count replay changed from {count} to {replay.count_value}"
            )
        return replay

    def trace(self, output_token_ids: Sequence[int]) -> dict[str, Any]:
        replay = self.replay(output_token_ids)
        trace: list[dict[str, Any]] = []
        for item in replay.traces:
            trace.append(item.as_dict())
            if item.marker == "<|num_nodule|>":
                trace.append({
                    "event": "dynamic_expansion",
                    "nodule_dictionaries": replay.count_value,
                })
        return {
            "count": replay.count_value,
            "program_complete": replay.program_complete,
            "next_action": replay.action.kind,
            "trace": trace,
        }


_RUNTIME_CACHE: dict[tuple[Any, ...], DynamicTemplateRuntime] = {}
_RUNTIME_CACHE_LOCK = threading.RLock()


def _runtime_cache_key(tokenizer_dir: str, kwargs: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(Path(tokenizer_dir).expanduser().resolve()),
        kwargs["template_style"],
        kwargs["include_json_tags"],
        kwargs["json_begin_tag"],
        kwargs["json_end_tag"],
        kwargs["json_indent"],
        kwargs["legacy_compat"],
        kwargs["template_add_special_tokens"],
        kwargs["max_dynamic_nodules"],
    )


def get_cached_runtime(tokenizer_dir: str, **kwargs: Any) -> DynamicTemplateRuntime:
    key = _runtime_cache_key(tokenizer_dir, kwargs)
    with _RUNTIME_CACHE_LOCK:
        runtime = _RUNTIME_CACHE.get(key)
        if runtime is None:
            runtime = DynamicTemplateRuntime.from_pretrained(
                tokenizer_dir,
                **kwargs,
            )
            _RUNTIME_CACHE[key] = runtime
        return runtime


class DynamicTemplateLogitsProcessor:
    """Stock-vLLM V0 custom logits processor for corrected template DC."""

    def __init__(
        self,
        tokenizer_dir: str,
        template_style: str = "canonical",
        include_json_tags: bool = True,
        json_begin_tag: str = "<json>",
        json_end_tag: str = "</json>",
        json_indent: int = 4,
        legacy_compat: bool = True,
        template_add_special_tokens: bool = False,
        max_dynamic_nodules: Optional[int] = 49,
        legacy_temperature: float = 1.0,
        legacy_top_p: float = 0.9,
        legacy_top_k: int = -1,
        legacy_min_p: float = 0.0,
        legacy_presence_penalty: float = 0.0,
        legacy_frequency_penalty: float = 0.0,
        legacy_repetition_penalty: float = 1.0,
    ) -> None:
        if max_dynamic_nodules is not None and max_dynamic_nodules < 0:
            max_dynamic_nodules = None
        if legacy_temperature <= 0:
            raise ValueError("legacy_temperature must be > 0")
        if not 0 < legacy_top_p <= 1:
            raise ValueError("legacy_top_p must be in (0, 1]")
        if legacy_top_k == 0 or legacy_top_k < -1:
            raise ValueError("legacy_top_k must be -1 or >= 1")
        if not 0 <= legacy_min_p <= 1:
            raise ValueError("legacy_min_p must be in [0, 1]")
        if legacy_repetition_penalty <= 0:
            raise ValueError("legacy_repetition_penalty must be > 0")

        self.tokenizer_dir = tokenizer_dir
        self.runtime_kwargs = {
            "template_style": template_style,
            "include_json_tags": bool(include_json_tags),
            "json_begin_tag": json_begin_tag,
            "json_end_tag": json_end_tag,
            "json_indent": int(json_indent),
            "legacy_compat": bool(legacy_compat),
            "template_add_special_tokens": bool(template_add_special_tokens),
            "max_dynamic_nodules": max_dynamic_nodules,
        }
        self.legacy_temperature = float(legacy_temperature)
        self.legacy_top_p = float(legacy_top_p)
        self.legacy_top_k = int(legacy_top_k)
        self.legacy_min_p = float(legacy_min_p)
        self.legacy_presence_penalty = float(legacy_presence_penalty)
        self.legacy_frequency_penalty = float(legacy_frequency_penalty)
        self.legacy_repetition_penalty = float(legacy_repetition_penalty)

    def clone(self):
        # Per-sequence state is reconstructed from output IDs; immutable config
        # and the process-wide immutable runtime cache are safe to share.
        return self

    def _runtime(self) -> DynamicTemplateRuntime:
        return get_cached_runtime(self.tokenizer_dir, **self.runtime_kwargs)

    @staticmethod
    def _apply_penalties(
        logits: Any,
        prompt_token_ids: Sequence[int],
        output_token_ids: Sequence[int],
        presence: float,
        frequency: float,
        repetition: float,
    ) -> Any:
        import torch

        vocab = logits.numel()
        prompt = [int(x) for x in prompt_token_ids if 0 <= int(x) < vocab]
        output = [int(x) for x in output_token_ids if 0 <= int(x) < vocab]
        if repetition != 1.0 and (prompt or output):
            used = torch.zeros(vocab, dtype=torch.bool, device=logits.device)
            if prompt:
                used[torch.tensor(prompt, device=logits.device)] = True
            if output:
                used[torch.tensor(output, device=logits.device)] = True
            penalties = torch.ones_like(logits)
            penalties[used] = repetition
            logits = torch.where(logits > 0, logits / penalties, logits * penalties)
        if output and (presence != 0.0 or frequency != 0.0):
            ids = torch.tensor(output, dtype=torch.long, device=logits.device)
            counts = torch.bincount(ids, minlength=vocab).to(logits.dtype)
            if frequency != 0.0:
                logits = logits - frequency * counts
            if presence != 0.0:
                logits = logits - presence * (counts > 0).to(logits.dtype)
        return logits

    @staticmethod
    def _apply_top_k_top_p(logits: Any, top_p: float, top_k: int) -> Any:
        import torch

        logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
        vocab = logits_sort.numel()
        k = vocab if top_k == -1 else min(top_k, vocab)
        threshold = logits_sort[vocab - k]
        logits_sort = logits_sort.masked_fill(logits_sort < threshold, -math.inf)
        probs_sort = torch.softmax(logits_sort, dim=-1)
        probs_sum = probs_sort.cumsum(dim=-1)
        top_p_mask = probs_sum <= (1.0 - top_p)
        top_p_mask[-1] = False
        logits_sort = logits_sort.masked_fill(top_p_mask, -math.inf)
        restored = torch.empty_like(logits_sort)
        restored.scatter_(0, logits_idx, logits_sort)
        return restored

    @staticmethod
    def _apply_min_p(logits: Any, min_p: float) -> Any:
        import torch

        probs = torch.softmax(logits, dim=-1)
        scaled_min = min_p * probs.max()
        return logits.masked_fill(probs < scaled_min, -math.inf)

    def _legacy_transforms(
        self,
        scores: Any,
        prompt_token_ids: Sequence[int],
        output_token_ids: Sequence[int],
    ) -> Any:
        import torch

        logits = scores.to(dtype=torch.float32).clone()
        logits = self._apply_penalties(
            logits,
            prompt_token_ids,
            output_token_ids,
            self.legacy_presence_penalty,
            self.legacy_frequency_penalty,
            self.legacy_repetition_penalty,
        )
        logits.div_(self.legacy_temperature)
        return logits

    def _select_allowed(self, transformed: Any, allowed: Sequence[int]) -> int:
        """Select after filtering within, never before, the legal token set."""

        import torch

        allowed_tensor = torch.tensor(
            list(allowed),
            dtype=torch.long,
            device=transformed.device,
        )
        local_logits = transformed[allowed_tensor]
        if self.legacy_top_p < 1.0 or self.legacy_top_k != -1:
            local_logits = self._apply_top_k_top_p(
                local_logits,
                self.legacy_top_p,
                self.legacy_top_k,
            )
        if self.legacy_min_p > 0:
            local_logits = self._apply_min_p(
                local_logits,
                self.legacy_min_p,
            )
        local_index = int(torch.argmax(local_logits).item())
        return int(allowed[local_index])

    @staticmethod
    def _force(scores: Any, token_id: int) -> Any:
        if token_id < 0 or token_id >= scores.numel():
            raise TemplateStateError(
                f"Forced token {token_id} is outside logits vocabulary "
                f"size {scores.numel()}"
            )
        scores.fill_(-math.inf)
        scores[token_id] = 0.0
        return scores

    def __call__(
        self,
        prompt_token_ids: list[int],
        output_token_ids: list[int],
        scores: Any,
    ) -> Any:
        runtime = self._runtime()
        replay = TemplateStateMachine(runtime).replay(output_token_ids)
        action = replay.action

        if action.kind in {"fixed", "eos"}:
            assert action.token_id is not None
            return self._force(scores, action.token_id)
        if action.kind != "candidate":
            raise TemplateStateError(f"Unexpected action: {action.kind}")

        transformed = self._legacy_transforms(
            scores,
            prompt_token_ids,
            output_token_ids,
        )
        if action.restrict_to_allowed:
            allowed = list(action.allowed_token_ids)
            if not allowed:
                raise TemplateStateError(
                    f"No allowed tokens for slot {action.slot_label}"
                )
            chosen = self._select_allowed(transformed, allowed)
        else:
            raise TemplateStateError(
                "Unrestricted candidate selection is disabled in the "
                "corrected constraint engine"
            )
        return self._force(scores, chosen)


def processor_kwargs(
    tokenizer_dir: str,
    *,
    template_style: str = "canonical",
    include_json_tags: bool = True,
    json_begin_tag: str = "<json>",
    json_end_tag: str = "</json>",
    json_indent: int = 4,
    legacy_compat: bool = True,
    template_add_special_tokens: bool = False,
    max_dynamic_nodules: Optional[int] = 49,
    legacy_temperature: float = 1.0,
    legacy_top_p: float = 0.9,
    legacy_top_k: int = -1,
    legacy_min_p: float = 0.0,
    legacy_presence_penalty: float = 0.0,
    legacy_frequency_penalty: float = 0.0,
    legacy_repetition_penalty: float = 1.0,
) -> dict[str, Any]:
    return {
        "tokenizer_dir": tokenizer_dir,
        "template_style": template_style,
        "include_json_tags": include_json_tags,
        "json_begin_tag": json_begin_tag,
        "json_end_tag": json_end_tag,
        "json_indent": json_indent,
        "legacy_compat": legacy_compat,
        "template_add_special_tokens": template_add_special_tokens,
        "max_dynamic_nodules": max_dynamic_nodules,
        "legacy_temperature": legacy_temperature,
        "legacy_top_p": legacy_top_p,
        "legacy_top_k": legacy_top_k,
        "legacy_min_p": legacy_min_p,
        "legacy_presence_penalty": legacy_presence_penalty,
        "legacy_frequency_penalty": legacy_frequency_penalty,
        "legacy_repetition_penalty": legacy_repetition_penalty,
    }


def canonicalize_legacy_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            LEGACY_TO_CANONICAL_KEYS.get(str(key), str(key)):
            canonicalize_legacy_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [canonicalize_legacy_keys(item) for item in value]
    return value


def coerce_canonical_types(value: Any, field_name: Optional[str] = None) -> Any:
    if isinstance(value, dict):
        return {
            key: coerce_canonical_types(item, str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [coerce_canonical_types(item, field_name) for item in value]
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if field_name in INTEGER_FIELDS:
        try:
            return int(stripped)
        except ValueError:
            return value
    if field_name in FLOAT_FIELDS:
        try:
            return float(stripped)
        except ValueError:
            return value
    return value


def validate_dense_prediction(
    prediction: Any,
    *,
    template_style: str = "canonical",
    raw_text: Optional[str] = None,
) -> dict[str, Any]:
    errors: list[str] = []
    obj = prediction
    if template_style == "legacy_snake_case":
        obj = canonicalize_legacy_keys(obj)
    if not isinstance(obj, dict):
        errors.append("parsed prediction is not a JSON object")
        return {"valid": False, "errors": errors}

    if raw_text and any(marker in raw_text for marker in CANONICAL_KEY_BY_MARKER):
        errors.append("placeholder marker leaked into final output")

    def validate_field(field_name: str, raw_value: Any, location: str) -> None:
        marker = MARKER_BY_CANONICAL_KEY[field_name]
        text = str(raw_value).strip()
        source_values = legacy_candidate_values()[marker]
        if len(source_values) >= 50:
            valid, complete = _flat_slot_status(marker, text)
            if not valid or not complete:
                errors.append(
                    f"{location} has invalid {field_name} value: {raw_value!r}"
                )
            return
        allowed = {
            str(value) for value in source_values
            if value not in CONTROL_CANDIDATES
        }
        if text not in allowed:
            errors.append(
                f"{location} has out-of-candidate {field_name} value: "
                f"{raw_value!r}"
            )

    for field_name in CANONICAL_TOP_LEVEL_KEYS:
        if field_name == "Nodules":
            continue
        if field_name in obj:
            validate_field(field_name, obj[field_name], "top level")

    raw_count = obj.get("Number of Nodules")
    try:
        count = int(str(raw_count).strip())
    except Exception:
        count = None
        errors.append(f"Number of Nodules is not integer-like: {raw_count!r}")

    nodules = obj.get("Nodules")
    if count == 0:
        if "Nodules" in obj:
            errors.append("Nodules key must be omitted when count is zero")
    elif count is not None:
        if not isinstance(nodules, list):
            errors.append("Nodules must be a list when count is positive")
        elif len(nodules) != count:
            errors.append(
                f"Number of Nodules={count}, but len(Nodules)={len(nodules)}"
            )

    expected_top = ["Number of Nodules"]
    if count is not None and count > 0:
        expected_top.append("Nodules")
    expected_top.extend(CANONICAL_TOP_LEVEL_KEYS[2:])
    if list(obj.keys()) != expected_top:
        errors.append(
            f"top-level keys/order differ: expected {expected_top}, "
            f"got {list(obj.keys())}"
        )

    if isinstance(nodules, list):
        for index, nodule in enumerate(nodules):
            if not isinstance(nodule, dict):
                errors.append(f"Nodule[{index}] is not an object")
                continue
            if tuple(nodule.keys()) != CANONICAL_NODULE_KEYS:
                errors.append(
                    f"Nodule[{index}] keys/order differ from dense template"
                )
            for field_name in CANONICAL_NODULE_KEYS:
                if field_name in nodule:
                    validate_field(
                        field_name,
                        nodule[field_name],
                        f"Nodule[{index}]",
                    )

    return {
        "valid": not errors,
        "errors": errors,
        "number_of_nodules": count,
        "nodule_array_length": len(nodules) if isinstance(nodules, list) else None,
    }


def format_trace(trace_payload: Mapping[str, Any]) -> str:
    lines: list[str] = []
    for item in trace_payload.get("trace", []):
        if item.get("event") == "dynamic_expansion":
            lines.append("[dynamic expansion]")
            lines.append(
                f"nodule dictionaries = {item.get('nodule_dictionaries')}"
            )
            continue
        lines.append(f"[slot] {item.get('slot')}")
        if item.get("allowed_values"):
            lines.append("allowed = " + ", ".join(item["allowed_values"]))
        else:
            lines.append(
                f"mode = {item.get('mode')}; candidates = "
                f"{item.get('candidate_count')}"
            )
        lines.append(f"chosen = {item.get('chosen')}")
    lines.append(f"program complete = {trace_payload.get('program_complete')}")
    return "\n".join(lines)


def load_runtime_for_audit(tokenizer_dir: str, **kwargs: Any) -> DynamicTemplateRuntime:
    return DynamicTemplateRuntime.from_pretrained(tokenizer_dir, **kwargs)
