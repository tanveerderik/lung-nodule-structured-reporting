#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:00:14 2026

@author: derik
"""

#!/usr/bin/env python3
import argparse
import os
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from accelerate import PartialState

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from peft.utils.other import fsdp_auto_wrap_policy

import torch


def build_messages(example):
    instruction = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    output = (example.get("output") or "").strip()

    user_text = instruction
    if inp:
        user_text += "\n\n" + inp

    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": output},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file", default=None)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
    )

    args = parser.parse_args()

    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file

    ext = os.path.splitext(args.train_file)[1].lower().replace(".", "")
    if ext == "jsonl":
        ext = "json"

    ds = load_dataset(ext, data_files=data_files)

    original_cols = ds["train"].column_names
    ds = ds.map(build_messages, remove_columns=original_cols)

    distributed_state = PartialState()
    torch.cuda.set_device(distributed_state.local_process_index)

    print(
        f"rank={distributed_state.process_index}, "
        f"local_rank={distributed_state.local_process_index}, "
        f"cuda_device={torch.cuda.current_device()}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,   # <-- the critical line for FSDP sharding
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )



    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )

    model.config.use_cache = False

    train_args = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        optim=args.optim,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=True,
        packing=args.packing,
        save_safetensors=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps,
        eval_strategy="steps" if args.eval_file else "no",
        eval_steps=args.eval_steps if args.eval_file else None,
        report_to=["wandb", "tensorboard"],
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_only_model=True,
    )


    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.model.print_trainable_parameters()

    fsdp_plugin = getattr(trainer.accelerator.state, "fsdp_plugin", None)
    if fsdp_plugin is not None:
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)


    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type(
            "FULL_STATE_DICT"
        )

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
