#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer


DEFAULT_MODELS = [
    "llama3_2_1B",
    "gemma3_4B",
    "mistral_7B",
    "qwen2_5_7B",
    "llama3_1_8B",
    "llama3_1_70B"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure tokenizer-specific prompt lengths and recommend "
            "a vLLM max_model_len for each model."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Project root directory.",
    )

    parser.add_argument(
        "--model-subdir",
        default="base",
        help=(
            "Model path relative to ROOT/models/<model_name>. "
            "Examples: base, sft/run_001_merged."
        ),
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Base-model or SFT-model evaluation JSON or JSONL file.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model directory names under ROOT/models.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1536,
        help="Generation allowance used during evaluation.",
    )

    parser.add_argument(
        "--buffer-tokens",
        type=int,
        default=128,
        help="Additional safety margin beyond prompt + completion.",
    )

    parser.add_argument(
        "--round-to",
        type=int,
        default=1024,
        help="Round recommended max_model_len upward to this boundary.",
    )

    parser.add_argument(
        "--output-json",
        required=True,
        help="Destination JSON summary.",
    )

    return parser.parse_args()


def load_any_dataset(path: str):
    extension = os.path.splitext(path)[1].lower().replace(".", "")

    if extension == "jsonl":
        extension = "json"

    if extension not in {"json", "csv", "parquet"}:
        raise ValueError(f"Unsupported dataset extension: {extension}")

    return load_dataset(
        extension,
        data_files={"data": path},
    )["data"]


def build_messages(example: dict[str, Any]):
    instruction = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    output = (example.get("output") or "").strip()

    user_text = instruction
    if inp:
        user_text += "\n\n" + inp

    return (
        [{"role": "user", "content": user_text}],
        output,
    )


def valid_context_value(value: Any) -> Optional[int]:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None

    # Tokenizers sometimes use enormous sentinel values to represent
    # "not explicitly specified." Do not treat those as real limits.
    if value <= 0 or value >= 10_000_000:
        return None

    return value


def extract_context_fields(config, tokenizer):
    fields = {}

    objects = [
        ("text_config", getattr(config, "text_config", None)),
        ("config", config),
    ]

    attributes = [
        "max_position_embeddings",
        "max_sequence_length",
        "seq_length",
        "n_positions",
    ]

    for object_name, obj in objects:
        if obj is None:
            continue

        for attribute in attributes:
            value = valid_context_value(getattr(obj, attribute, None))
            if value is not None:
                fields[f"{object_name}.{attribute}"] = value

    tokenizer_limit = valid_context_value(
        getattr(tokenizer, "model_max_length", None)
    )

    if tokenizer_limit is not None:
        fields["tokenizer.model_max_length"] = tokenizer_limit

    # Prefer the text-model max_position_embeddings field when present.
    preferred_keys = [
        "text_config.max_position_embeddings",
        "config.max_position_embeddings",
        "text_config.max_sequence_length",
        "config.max_sequence_length",
        "text_config.seq_length",
        "config.seq_length",
        "text_config.n_positions",
        "config.n_positions",
        "tokenizer.model_max_length",
    ]

    native_context = None

    for key in preferred_keys:
        if key in fields:
            native_context = fields[key]
            break

    return native_context, fields


def round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("--round-to must be positive")

    return int(math.ceil(value / multiple) * multiple)


def percentile(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q))


def analyze_model(
    model_name: str,
    model_dir: Path,
    dataset,
    max_new_tokens: int,
    buffer_tokens: int,
    round_to: int,
):
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print()
    print("=" * 80)
    print(f"MODEL: {model_name}")
    print(f"PATH:  {model_dir}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )

    config = AutoConfig.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )

    prompt_lengths = []
    output_lengths = []
    full_lengths = []

    for example in dataset:
        prompt_messages, output_text = build_messages(example)

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_messages = prompt_messages + [
            {"role": "assistant", "content": output_text}
        ]

        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_length = len(
            tokenizer(
                prompt_text,
                add_special_tokens=False,
            )["input_ids"]
        )

        output_length = len(
            tokenizer(
                output_text,
                add_special_tokens=False,
            )["input_ids"]
        )

        full_length = len(
            tokenizer(
                full_text,
                add_special_tokens=False,
            )["input_ids"]
        )

        prompt_lengths.append(prompt_length)
        output_lengths.append(output_length)
        full_lengths.append(full_length)

    prompt_lengths = np.asarray(prompt_lengths, dtype=np.int64)
    output_lengths = np.asarray(output_lengths, dtype=np.int64)
    full_lengths = np.asarray(full_lengths, dtype=np.int64)

    maximum_prompt = int(prompt_lengths.max())
    maximum_output = int(output_lengths.max())

    required_context = (
        maximum_prompt
        + max_new_tokens
        + buffer_tokens
    )

    recommended_context = round_up(
        required_context,
        round_to,
    )

    native_context, context_fields = extract_context_fields(
        config,
        tokenizer,
    )

    fits_native_context = (
        native_context is None
        or recommended_context <= native_context
    )

    completion_cap_covers_gold = maximum_output <= max_new_tokens

    result = {
        "model_name": model_name,
        "model_dir": str(model_dir),
        "samples": int(len(prompt_lengths)),
        "prompt_tokens": {
            "min": int(prompt_lengths.min()),
            "mean": float(prompt_lengths.mean()),
            "median": float(np.median(prompt_lengths)),
            "p90": percentile(prompt_lengths, 90),
            "p95": percentile(prompt_lengths, 95),
            "p99": percentile(prompt_lengths, 99),
            "max": maximum_prompt,
        },
        "gold_output_tokens": {
            "min": int(output_lengths.min()),
            "mean": float(output_lengths.mean()),
            "median": float(np.median(output_lengths)),
            "p90": percentile(output_lengths, 90),
            "p95": percentile(output_lengths, 95),
            "p99": percentile(output_lengths, 99),
            "max": maximum_output,
        },
        "full_gold_chat_tokens": {
            "min": int(full_lengths.min()),
            "mean": float(full_lengths.mean()),
            "median": float(np.median(full_lengths)),
            "p90": percentile(full_lengths, 90),
            "p95": percentile(full_lengths, 95),
            "p99": percentile(full_lengths, 99),
            "max": int(full_lengths.max()),
        },
        "max_new_tokens": max_new_tokens,
        "buffer_tokens": buffer_tokens,
        "required_context": required_context,
        "recommended_max_model_len": recommended_context,
        "native_context": native_context,
        "context_fields": context_fields,
        "fits_native_context": fits_native_context,
        "completion_cap_covers_gold": completion_cap_covers_gold,
    }

    print(f"Samples:                     {len(prompt_lengths):,}")
    print(f"Maximum prompt:              {maximum_prompt:,}")
    print(f"P99 prompt:                  {percentile(prompt_lengths, 99):,.1f}")
    print(f"Maximum gold output:         {maximum_output:,}")
    print(f"Generation allowance:        {max_new_tokens:,}")
    print(f"Required with buffer:        {required_context:,}")
    print(f"Recommended max_model_len:   {recommended_context:,}")
    print(f"Configured native context:   {native_context}")
    print(f"Fits native context:         {fits_native_context}")
    print(f"Gold output fits generation: {completion_cap_covers_gold}")

    if context_fields:
        print("Detected context fields:")
        for key, value in context_fields.items():
            print(f"  {key}: {value:,}")

    return result


def main():
    args = parse_args()

    root = Path(args.root).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = load_any_dataset(str(dataset_path))

    results = {}

    for model_name in args.models:
        model_dir = root / "models" / model_name / args.model_subdir

        results[model_name] = analyze_model(
            model_name=model_name,
            model_dir=model_dir,
            dataset=dataset,
            max_new_tokens=args.max_new_tokens,
            buffer_tokens=args.buffer_tokens,
            round_to=args.round_to,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print()
    print("=" * 80)
    print("CONTEXT AUDIT SUMMARY")
    print("=" * 80)

    header = (
        f"{'Model':<18}"
        f"{'Prompt max':>13}"
        f"{'Gold max':>11}"
        f"{'Recommended':>14}"
        f"{'Native':>12}"
        f"{'Fits':>8}"
    )
    print(header)
    print("-" * len(header))

    for model_name, result in results.items():
        native = result["native_context"]
        native_text = f"{native:,}" if native is not None else "unknown"

        print(
            f"{model_name:<18}"
            f"{result['prompt_tokens']['max']:>13,}"
            f"{result['gold_output_tokens']['max']:>11,}"
            f"{result['recommended_max_model_len']:>14,}"
            f"{native_text:>12}"
            f"{str(result['fits_native_context']):>8}"
        )

    print()
    print(f"Saved audit: {output_path}")


if __name__ == "__main__":
    main()
