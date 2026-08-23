#!/usr/bin/env python3
"""CPU tokenizer/program audit and offline trace for authentic template DC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dynamic_template_constraint import (
    CANONICAL_KEY_BY_MARKER,
    CANONICAL_NODULE_KEYS,
    CANONICAL_TOP_LEVEL_KEYS,
    CONTROL_CANDIDATES,
    TemplateStateMachine,
    format_trace,
    legacy_candidate_values,
    load_runtime_for_audit,
    validate_dense_prediction,
)


def load_json(path: str) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_template_audit(path: str) -> dict[str, Any]:
    data = load_json(path)
    findings = (
        data.get("Lungs Pleura", {})
        .get("Nodule Findings")
    )
    errors: list[str] = []
    if not isinstance(findings, dict):
        return {
            "path": str(Path(path).expanduser().resolve()),
            "valid": False,
            "errors": ["Missing Lungs Pleura.Nodule Findings object"],
        }

    if tuple(findings.keys()) != CANONICAL_TOP_LEVEL_KEYS:
        errors.append(
            "Nodule Findings keys/order differ from the canonical dense template"
        )
    nodules = findings.get("Nodules")
    if not (
        isinstance(nodules, list)
        and len(nodules) == 1
        and isinstance(nodules[0], dict)
    ):
        errors.append("Nodules must contain exactly one schema dictionary")
        nodule_schema = {}
    else:
        nodule_schema = nodules[0]
        if tuple(nodule_schema.keys()) != CANONICAL_NODULE_KEYS:
            errors.append("Nodule schema keys/order differ from the dense template")

    candidate_source = legacy_candidate_values()
    enum_mismatches = []
    for marker, canonical_key in CANONICAL_KEY_BY_MARKER.items():
        if canonical_key in nodule_schema:
            field = nodule_schema[canonical_key]
        else:
            field = findings.get(canonical_key)
        if not isinstance(field, dict):
            continue
        values = field.get("values")
        if not isinstance(values, list):
            continue
        # Range/date descriptors in the canonical schema are provenance hints,
        # not literal legacy candidates. Exact expanded behavior is audited by
        # the runtime rows instead.
        if any(
            isinstance(value, str)
            and ("-" in value and any(ch.isdigit() for ch in value))
            for value in values
        ) or any(value == "MM/DD/YYYY" for value in values):
            continue
        canonical = {str(value) for value in values}
        legacy = {
            str(value)
            for value in candidate_source[marker]
            if value not in CONTROL_CANDIDATES
        }
        if canonical != legacy:
            enum_mismatches.append({
                "marker": marker,
                "field": canonical_key,
                "canonical_only": sorted(canonical - legacy),
                "legacy_only": sorted(legacy - canonical),
            })
    if enum_mismatches:
        errors.append("Categorical candidates differ from the legacy definition")

    return {
        "path": str(Path(path).expanduser().resolve()),
        "valid": not errors,
        "errors": errors,
        "enum_mismatches": enum_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--template_file")
    parser.add_argument(
        "--template_style",
        choices=["canonical", "legacy_snake_case"],
        default="canonical",
    )
    parser.add_argument(
        "--include_json_tags",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--json_begin_tag", default="<json>")
    parser.add_argument("--json_end_tag", default="</json>")
    parser.add_argument("--json_indent", type=int, default=4)
    parser.add_argument(
        "--legacy_compat",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--template_add_special_tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max_dynamic_nodules", type=int, default=49)
    parser.add_argument(
        "--trace_token_ids_json",
        help="Optional JSON file containing one generated token-ID list.",
    )
    parser.add_argument(
        "--trace_text",
        help=(
            "Optional raw output text to re-encode and trace. Exact token-ID "
            "input is preferred for legacy flattened sequences."
        ),
    )
    args = parser.parse_args()

    max_dynamic_nodules = (
        None if args.max_dynamic_nodules < 0 else args.max_dynamic_nodules
    )
    runtime = load_runtime_for_audit(
        args.tokenizer_dir,
        template_style=args.template_style,
        include_json_tags=args.include_json_tags,
        json_begin_tag=args.json_begin_tag,
        json_end_tag=args.json_end_tag,
        json_indent=args.json_indent,
        legacy_compat=args.legacy_compat,
        template_add_special_tokens=args.template_add_special_tokens,
        max_dynamic_nodules=max_dynamic_nodules,
    )

    report = runtime.audit()
    report["tokenizer_dir"] = str(
        Path(args.tokenizer_dir).expanduser().resolve()
    )
    report["canonical_template_audit"] = (
        canonical_template_audit(args.template_file)
        if args.template_file else None
    )

    sanity = []
    for count in (0, 1, 2):
        tokens = runtime.render_tokens(count)
        text = runtime.decode_output(tokens)
        try:
            parsed = json.loads(
                text.split("\n", 1)[1].rsplit("\n", 1)[0]
                if args.include_json_tags else text
            )
            validation = validate_dense_prediction(
                parsed,
                template_style=args.template_style,
                raw_text=text,
            )
        except Exception as exc:
            validation = {"valid": False, "errors": [str(exc)]}
        replay = TemplateStateMachine(runtime).replay(tokens)
        sanity.append({
            "count": count,
            "generated_tokens": len(tokens),
            "program_complete": replay.program_complete,
            "next_action": replay.action.kind,
            "dense_validation": validation,
        })
    report["count_sanity"] = sanity

    trace_ids = None
    trace_source = None
    if args.trace_token_ids_json:
        trace_ids = load_json(args.trace_token_ids_json)
        trace_source = "exact_token_ids"
    elif args.trace_text:
        trace_ids = runtime.encode(args.trace_text, add_special_tokens=False)
        trace_source = "reencoded_text"
    if trace_ids is not None:
        if not isinstance(trace_ids, list) or not all(
            isinstance(item, int) for item in trace_ids
        ):
            raise SystemExit("Trace token input must be a JSON integer list")
        machine = TemplateStateMachine(runtime)
        try:
            trace_payload = machine.trace(trace_ids)
        except Exception as full_trace_error:
            # In the canonical stock-server mode EOS is forced after the
            # compiled visible template, so it is present in token logprobs but
            # is not a program token.  Historical parity mode may compile EOS
            # into the template itself; retain it when the full trace succeeds.
            if not trace_ids or trace_ids[-1] != runtime.eos_token_id:
                raise
            try:
                trace_ids = trace_ids[:-1]
                trace_payload = machine.trace(trace_ids)
            except Exception:
                raise full_trace_error
        report["trace_source"] = trace_source
        report["trace"] = trace_payload

    report["valid"] = bool(
        report["output_ids_within_original_vocab"]
        and all(
            item["program_complete"]
            and item["next_action"] == "eos"
            and item["dense_validation"]["valid"]
            for item in sanity
        )
        and (
            report["canonical_template_audit"] is None
            or report["canonical_template_audit"]["valid"]
        )
    )

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report.get("trace"):
        print("\n" + format_trace(report["trace"]))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
