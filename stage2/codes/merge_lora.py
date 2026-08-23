#!/usr/bin/env python3
"""Merge a PEFT LoRA/QLoRA adapter into an unquantized base model."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import peft
import torch
import transformers
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a PEFT LoRA/QLoRA adapter into its unquantized base model "
            "and save a standalone Safetensors checkpoint."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--adapter-model",
        required=True,
        help="Local adapter directory or Hugging Face adapter model ID.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory in which to save the merged standalone model.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help=(
            "Local base-model directory or Hugging Face model ID. When omitted, "
            "base_model_name_or_path is read from the adapter configuration."
        ),
    )
    parser.add_argument(
        "--tokenizer-source",
        default=None,
        help=(
            "Tokenizer directory/model ID. When omitted, the script first tries "
            "the adapter and then falls back to the base model."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32", "auto"),
        default="bfloat16",
        help="Data type used to load and save the full merged model.",
    )
    parser.add_argument(
        "--device-map",
        choices=("auto", "cpu", "cuda:0", "none"),
        default="auto",
        help=(
            "Model placement. Use auto for multi-GPU/CPU dispatch, cpu for a "
            "CPU-only merge, cuda:0 for one visible GPU, or none for default loading."
        ),
    )
    parser.add_argument(
        "--max-memory",
        action="append",
        default=[],
        metavar="DEVICE=LIMIT",
        help=(
            "Optional Accelerate memory limit; repeat as needed, for example "
            "--max-memory 0=76GiB --max-memory 1=76GiB --max-memory cpu=200GiB."
        ),
    )
    parser.add_argument(
        "--offload-folder",
        default=None,
        help="Folder for disk offload when the automatic device map requires it.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum size of each output Safetensors shard.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional base-model Hub revision.",
    )
    parser.add_argument(
        "--adapter-revision",
        default=None,
        help="Optional adapter Hub revision.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow custom model/tokenizer code from a model repository.",
    )
    parser.add_argument(
        "--safe-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check adapter weights for non-finite values while merging.",
    )
    parser.add_argument(
        "--low-cpu-mem-usage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the low-memory Transformers loading path.",
    )
    parser.add_argument(
        "--offload-state-dict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Temporarily offload the CPU state dictionary when necessary.",
    )
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Perform structural checks after saving.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete a non-empty output directory before saving.",
    )
    return parser


def parse_dtype(name: str) -> torch.dtype | str:
    mapping: dict[str, torch.dtype | str] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }
    return mapping[name]


def parse_device_map(name: str) -> str | dict[str, str] | None:
    if name == "auto":
        return "auto"
    if name == "cpu":
        return {"": "cpu"}
    if name == "cuda:0":
        if not torch.cuda.is_available():
            raise RuntimeError("--device-map cuda:0 was requested, but CUDA is unavailable.")
        return {"": "cuda:0"}
    if name == "none":
        return None
    raise ValueError(f"Unsupported device map: {name}")


def parse_max_memory(items: list[str]) -> dict[int | str, str] | None:
    if not items:
        return None

    result: dict[int | str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --max-memory value {item!r}; expected DEVICE=LIMIT."
            )
        raw_device, raw_limit = item.split("=", 1)
        device = raw_device.strip()
        limit = raw_limit.strip()
        if not device or not limit:
            raise ValueError(
                f"Invalid --max-memory value {item!r}; expected DEVICE=LIMIT."
            )
        key: int | str = int(device) if device.isdigit() else device
        result[key] = limit
    return result


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}\n"
                "Choose another directory or pass --overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def choose_tokenizer_source(
    requested: str | None,
    adapter_model: str,
    base_model: str,
    common_kwargs: dict[str, Any],
):
    candidates = [requested] if requested else [adapter_model, base_model]
    errors: list[str] = []

    for source in candidates:
        if source is None:
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, **common_kwargs)
            return tokenizer, source
        except Exception as exc:  # fallback is intentional
            errors.append(f"{source}: {type(exc).__name__}: {exc}")

    joined = "\n  - ".join(errors)
    raise RuntimeError(f"Unable to load a tokenizer from any candidate:\n  - {joined}")


def verify_output(output_dir: Path) -> None:
    config_path = output_dir / "config.json"
    single_file = output_dir / "model.safetensors"
    index_path = output_dir / "model.safetensors.index.json"

    if not config_path.is_file():
        raise RuntimeError(f"Missing saved config: {config_path}")

    if single_file.is_file():
        if single_file.stat().st_size == 0:
            raise RuntimeError(f"Saved model file is empty: {single_file}")
        print(f"Verified single Safetensors file: {single_file.name}")
        return

    if not index_path.is_file():
        raise RuntimeError(
            "No model.safetensors or model.safetensors.index.json was produced."
        )

    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"Invalid or empty weight_map in {index_path}")

    shards = sorted(set(weight_map.values()))
    missing = [name for name in shards if not (output_dir / name).is_file()]
    empty = [
        name
        for name in shards
        if (output_dir / name).is_file() and (output_dir / name).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"Missing output shards: {missing}")
    if empty:
        raise RuntimeError(f"Empty output shards: {empty}")

    print(f"Verified {len(weight_map):,} tensors across {len(shards)} shards.")


def main() -> int:
    args = build_parser().parse_args()

    adapter_model = args.adapter_model
    output_dir = Path(args.output_dir).expanduser().resolve()
    dtype = parse_dtype(args.dtype)
    device_map = parse_device_map(args.device_map)
    max_memory = parse_max_memory(args.max_memory)

    adapter_config = PeftConfig.from_pretrained(
        adapter_model,
        revision=args.adapter_revision,
        cache_dir=args.cache_dir,
    )
    base_model = args.base_model or adapter_config.base_model_name_or_path
    if not base_model:
        raise RuntimeError(
            "No base model was supplied and the adapter configuration does not "
            "contain base_model_name_or_path."
        )

    prepare_output_dir(output_dir, args.overwrite)

    if args.offload_folder:
        Path(args.offload_folder).expanduser().mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print(f"Base model       : {base_model}")
    print(f"Adapter          : {adapter_model}")
    print(f"Output           : {output_dir}")
    print(f"Dtype            : {args.dtype}")
    print(f"Device map       : {args.device_map}")
    print(f"Max memory       : {max_memory or 'automatic'}")
    print(f"Safe merge       : {args.safe_merge}")
    print(f"Transformers     : {transformers.__version__}")
    print(f"PEFT             : {peft.__version__}")
    print(f"PyTorch          : {torch.__version__}")
    print("=" * 88)

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
        "trust_remote_code": args.trust_remote_code,
        "revision": args.revision,
        "cache_dir": args.cache_dir,
    }
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
    if args.offload_folder:
        model_kwargs["offload_folder"] = str(
            Path(args.offload_folder).expanduser().resolve()
        )
        model_kwargs["offload_state_dict"] = args.offload_state_dict

    # No BitsAndBytesConfig or load_in_4bit/load_in_8bit is used here. QLoRA
    # adapters must be merged into an unquantized full-precision base model.
    print("\nLoading unquantized base model...")
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    print("Loading adapter...")
    peft_model = PeftModel.from_pretrained(
        model,
        adapter_model,
        revision=args.adapter_revision,
        cache_dir=args.cache_dir,
        is_trainable=False,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )

    print("Merging adapter into base weights...")
    merged_model = peft_model.merge_and_unload(
        safe_merge=args.safe_merge,
        progressbar=True,
    )

    if isinstance(dtype, torch.dtype):
        merged_model.config.torch_dtype = dtype

    print("Saving standalone Safetensors model...")
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    tokenizer_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "cache_dir": args.cache_dir,
    }
    tokenizer, tokenizer_source = choose_tokenizer_source(
        args.tokenizer_source,
        adapter_model,
        base_model,
        tokenizer_kwargs,
    )
    print(f"Saving tokenizer from: {tokenizer_source}")
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "adapter_model": adapter_model,
        "tokenizer_source": tokenizer_source,
        "dtype": args.dtype,
        "safe_merge": args.safe_merge,
        "max_shard_size": args.max_shard_size,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "torch_version": torch.__version__,
    }
    with (output_dir / "merge_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    if args.verify:
        print("Verifying saved files...")
        verify_output(output_dir)

    del tokenizer
    del merged_model
    del peft_model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nMerge complete: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
