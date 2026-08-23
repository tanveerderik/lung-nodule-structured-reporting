#!/usr/bin/env python3
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from transformers import AutoTokenizer
from tqdm import tqdm

from date_normalization import DATE_SCORING_POLICY
from dynamic_template_constraint import (
    PROCESSOR_QUALNAME,
    TemplateStateMachine,
    format_trace,
    get_cached_runtime,
    processor_kwargs,
    validate_dense_prediction,
)
from nodule_scoring import is_null, score_case
from prediction_normalization import normalize_prediction, safe_json_loads
from prompt_policy import summarize_instructions


def build_user_text(example):
    instruction = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    return instruction + ("\n\n" + inp if inp else "")


def load_test_dataset(path):
    ext = os.path.splitext(path)[1].lower().replace(".", "")
    if ext == "jsonl":
        ext = "json"
    return load_dataset(ext, data_files={"test": path})["test"]


def get_case_id(example, fallback_index):
    for key in ["id", "ID", "report_id", "Report ID", "accession_number", "Accession Number"]:
        if key in example and not is_null(example[key]):
            return example[key]
    return fallback_index


def build_prompt(tokenizer, user_text):
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_output_schema(path):
    with Path(path).open("r", encoding="utf-8") as f:
        schema = json.load(f)
    if not isinstance(schema, dict):
        raise ValueError("The output schema must be a JSON object")
    return schema


def build_v0_tagged_grammar(schema, begin_tag, end_tag, json_indent=4):
    """Build one GBNF grammar for begin tag + schema JSON + end tag.

    vLLM 0.8.5.post1 V0 supports XGrammar's `guided_grammar`, but its V0
    XGrammar backend does not implement `structural_tag`. Concatenating literal
    tags and the schema grammar gives V0 the equivalent *forced* tagged output
    for this single-structure use case. The exact training data uses a newline
    around JSON and json.dumps(..., indent=4) formatting.
    """
    try:
        import xgrammar as xgr
    except ImportError as exc:
        raise RuntimeError(
            "xgrammar is required for --decoding_mode v0_tagged_grammar; "
            "run this evaluator in the vLLM 0.8.5.post1 environment"
        ) from exc

    def literal_grammar(text):
        # json.dumps produces a valid quoted GBNF string and correctly escapes
        # the literal backslash in <\json>.
        return xgr.Grammar.from_ebnf(
            "root ::= " + json.dumps(text, ensure_ascii=False)
        )

    body = xgr.Grammar.from_json_schema(
        schema,
        any_whitespace=False,
        indent=json_indent,
        strict_mode=True,
    )
    return str(
        xgr.Grammar.concat(
            literal_grammar(begin_tag + "\n"),
            body,
            literal_grammar("\n" + end_tag),
        )
    )


def call_vllm(
    client,
    model_name,
    request_item,
    max_tokens,
    temperature,
    top_p=1.0,
    top_k=-1,
    min_p=0.0,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    repetition_penalty=1.0,
    seed=0,
    decoding_mode="unconstrained",
    guided_grammar=None,
    guided_decoding_backend="xgrammar:no-fallback",
    completion_add_special_tokens=None,
    dynamic_processor_config=None,
    dynamic_trace=False,
    retries=3,
):
    last_err = None

    for attempt in range(retries):
        try:
            extra_body = {}
            if decoding_mode == "v0_tagged_grammar":
                extra_body.update({
                    "guided_grammar": guided_grammar,
                    "guided_decoding_backend": guided_decoding_backend,
                })
            elif decoding_mode == "authentic_dynamic_template":
                if not dynamic_processor_config:
                    raise ValueError(
                        "authentic_dynamic_template requires processor config"
                    )
                extra_body.update({
                    "logits_processors": [{
                        "qualname": PROCESSOR_QUALNAME,
                        "kwargs": dynamic_processor_config,
                    }],
                    # The evaluator already rendered the complete chat prompt.
                    "add_special_tokens": False,
                    # All legacy transforms occur inside the custom processor.
                    # These stock transforms must remain neutral downstream.
                    "top_k": -1,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                })
                if dynamic_trace:
                    extra_body["return_tokens_as_token_ids"] = True
            # Both ordinary and XGrammar runs must use the same value. The
            # paired launcher explicitly passes false because the prompt has
            # already been rendered by apply_chat_template(). Preserve false
            # as the historical constrained default when the option is omitted.
            effective_add_special_tokens = completion_add_special_tokens
            if effective_add_special_tokens is None and decoding_mode in {
                "v0_tagged_grammar",
                "authentic_dynamic_template",
            }:
                effective_add_special_tokens = False
            if effective_add_special_tokens is not None:
                extra_body["add_special_tokens"] = effective_add_special_tokens

            if decoding_mode != "authentic_dynamic_template":
                # vLLM-specific sampling fields are made explicit so model
                # generation_config.json files cannot silently change one arm.
                extra_body.update({
                    "top_k": top_k,
                    "min_p": min_p,
                    "repetition_penalty": repetition_penalty,
                    "seed": seed,
                })

            kwargs = {
                "model": model_name,
                "prompt": request_item["prompt"],
                "max_tokens": max_tokens,
                "temperature": (
                    0.0
                    if decoding_mode == "authentic_dynamic_template"
                    else temperature
                ),
            }
            if decoding_mode == "authentic_dynamic_template":
                kwargs.update({
                    "top_p": 1.0,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                })
                if dynamic_trace:
                    kwargs["logprobs"] = 0
            else:
                kwargs.update({
                    "top_p": top_p,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty,
                })
            if extra_body:
                kwargs["extra_body"] = extra_body
            response = client.completions.create(**kwargs)
            choice = response.choices[0]
            text = choice.text or ""

            generated_token_ids = None
            if dynamic_trace and choice.logprobs is not None:
                token_strings = choice.logprobs.tokens or []
                generated_token_ids = []
                for token in token_strings:
                    if not isinstance(token, str) or not token.startswith("token_id:"):
                        raise ValueError(
                            "vLLM did not return token IDs in debug-trace mode"
                        )
                    generated_token_ids.append(int(token.split(":", 1)[1]))

            usage = response.usage

            return {
                # Keep the raw returned boundaries exactly as generated.  Do
                # not strip the closing tag or stop string.
                "text": text,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
                "generated_token_ids": generated_token_ids,
                "finish_reason": choice.finish_reason,
            }

        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(
        f"vLLM request failed after {retries} attempts: {last_err}"
    )


def summarize_token_counts(values):
    values = sorted(v for v in values if v is not None)

    if not values:
        return {
            "Total": 0,
            "Mean": None,
            "Median": None,
            "Min": None,
            "Max": None,
            "P95": None,
        }

    n = len(values)

    def percentile(p):
        index = round((n - 1) * p)
        return values[index]

    midpoint = n // 2
    if n % 2:
        median = values[midpoint]
    else:
        median = (values[midpoint - 1] + values[midpoint]) / 2

    return {
        "Total": sum(values),
        "Mean": sum(values) / n,
        "Median": median,
        "Min": values[0],
        "Max": values[-1],
        "P95": percentile(0.95),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--test_file", required=True)

    parser.add_argument("--base_url", default="http://localhost:8080/v1")
    parser.add_argument("--api_key", default="EMPTY")

    parser.add_argument("--output_json", default="sft_eval_vllm_cases.json")
    parser.add_argument("--summary_json", default="sft_eval_vllm_summary.json")

    parser.add_argument("--eval_context_limit", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min_similarity", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument(
        "--decoding_mode",
        choices=[
            "unconstrained",
            "v0_tagged_grammar",
            "authentic_dynamic_template",
        ],
        default="unconstrained",
        help=(
            "Unconstrained generation, the prior XGrammar ablation, or the "
            "authentic dynamic template-constrained decoder."
        ),
    )
    parser.add_argument(
        "--schema_file",
        help="Sparse JSON Schema used by V0 tagged-grammar decoding.",
    )
    parser.add_argument("--json_begin_tag", default="<json>")
    parser.add_argument(
        "--json_end_tag",
        help=(
            "Exact SFT closing tag, for example '<\\json>' or '</json>'. "
            "Required for constrained modes so the evaluator never guesses."
        ),
    )
    parser.add_argument(
        "--guided_decoding_backend",
        default="xgrammar:no-fallback",
        help="Per-request backend; no-fallback prevents silent backend changes.",
    )
    parser.add_argument(
        "--json_indent",
        type=int,
        default=4,
        help="Exact JSON indentation used in all audited SFT targets.",
    )
    parser.add_argument(
        "--completion_add_special_tokens",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override vLLM CompletionRequest.add_special_tokens. For a controlled "
            "ordinary/XGrammar comparison, pass the same explicit value to both "
            "runs. False is correct for this already chat-template-rendered prompt."
        ),
    )
    parser.add_argument(
        "--dynamic_template_style",
        choices=["canonical", "legacy_snake_case"],
        default="canonical",
        help=(
            "Use current human-readable keys, or the historical snake_case "
            "keys for old-fork parity tests."
        ),
    )
    parser.add_argument(
        "--dynamic_include_json_tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Force the configured JSON boundary tags around the dense template. "
            "Disable for the historical tagless parity mode."
        ),
    )
    parser.add_argument(
        "--dynamic_legacy_compat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the corrected typed-composition mode for original >=50 "
            "candidate slots. Disable to require exact full-candidate tries."
        ),
    )
    parser.add_argument(
        "--dynamic_template_add_special_tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether the private compiler tokenizer adds BOS/EOS around the "
            "template. False is the valid stock-server output mode."
        ),
    )
    parser.add_argument(
        "--dynamic_max_nodules",
        type=int,
        default=49,
        help=(
            "Explicit safety cap after legacy count decoding; -1 removes it for "
            "exact behavior outside the intended 0..49 domain."
        ),
    )
    parser.add_argument("--dynamic_legacy_temperature", type=float, default=1.0)
    parser.add_argument("--dynamic_legacy_top_p", type=float, default=0.9)
    parser.add_argument("--dynamic_legacy_top_k", type=int, default=-1)
    parser.add_argument("--dynamic_legacy_min_p", type=float, default=0.0)
    parser.add_argument("--dynamic_legacy_presence_penalty", type=float, default=0.0)
    parser.add_argument("--dynamic_legacy_frequency_penalty", type=float, default=0.0)
    parser.add_argument("--dynamic_legacy_repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--dynamic_trace",
        action="store_true",
        help=(
            "Request generated token IDs, reconstruct a slot-by-slot trace, and "
            "store it in each case record. Intended only for small sanity runs."
        ),
    )
    parser.add_argument(
        "--template_file",
        help=(
            "Canonical lungs_pleura_nodule_template.json recorded for provenance; "
            "candidate behavior remains the exact legacy candidate definition."
        ),
    )

    args = parser.parse_args()

    constrained = args.decoding_mode != "unconstrained"
    grammar_constrained = args.decoding_mode == "v0_tagged_grammar"
    dynamic_constrained = args.decoding_mode == "authentic_dynamic_template"
    if grammar_constrained and not args.schema_file:
        parser.error("--schema_file is required for v0_tagged_grammar")
    if (
        (grammar_constrained or (dynamic_constrained and args.dynamic_include_json_tags))
        and args.json_end_tag is None
    ):
        parser.error("--json_end_tag is required when forced JSON tags are enabled")
    if (
        args.decoding_mode == "v0_tagged_grammar"
        and args.guided_decoding_backend.split(":", 1)[0] != "xgrammar"
    ):
        parser.error("v0_tagged_grammar requires the xgrammar backend")
    if dynamic_constrained and args.dynamic_trace and args.limit is None:
        parser.error("--dynamic_trace requires --limit for a bounded debug run")
    output_schema = load_output_schema(args.schema_file) if grammar_constrained else None
    guided_grammar = None
    if args.decoding_mode == "v0_tagged_grammar":
        guided_grammar = build_v0_tagged_grammar(
            output_schema,
            args.json_begin_tag,
            args.json_end_tag,
            args.json_indent,
        )

    dynamic_processor_config = None
    dynamic_runtime_config = None
    if dynamic_constrained:
        max_dynamic_nodules = (
            None if args.dynamic_max_nodules < 0 else args.dynamic_max_nodules
        )
        dynamic_processor_config = processor_kwargs(
            args.tokenizer_dir,
            template_style=args.dynamic_template_style,
            include_json_tags=args.dynamic_include_json_tags,
            json_begin_tag=args.json_begin_tag,
            json_end_tag=args.json_end_tag or "",
            json_indent=args.json_indent,
            legacy_compat=args.dynamic_legacy_compat,
            template_add_special_tokens=args.dynamic_template_add_special_tokens,
            max_dynamic_nodules=max_dynamic_nodules,
            legacy_temperature=args.dynamic_legacy_temperature,
            legacy_top_p=args.dynamic_legacy_top_p,
            legacy_top_k=args.dynamic_legacy_top_k,
            legacy_min_p=args.dynamic_legacy_min_p,
            legacy_presence_penalty=args.dynamic_legacy_presence_penalty,
            legacy_frequency_penalty=args.dynamic_legacy_frequency_penalty,
            legacy_repetition_penalty=args.dynamic_legacy_repetition_penalty,
        )
        dynamic_runtime_config = {
            key: dynamic_processor_config[key]
            for key in (
                "template_style",
                "include_json_tags",
                "json_begin_tag",
                "json_end_tag",
                "json_indent",
                "legacy_compat",
                "template_add_special_tokens",
                "max_dynamic_nodules",
            )
        }

    ds = load_test_dataset(args.test_file)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir,
        trust_remote_code=True,
    )

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    examples = []
    request_items = []
    skipped_too_long = []

    max_prompt_tokens = args.eval_context_limit - args.max_new_tokens


    for i, ex in enumerate(ds):
        user_text = build_user_text(ex)
        prompt = build_prompt(tokenizer, user_text)

        prompt_tokens = len(
            tokenizer(
            prompt,
                add_special_tokens=False,
            )["input_ids"]
        )

        case_id = get_case_id(ex, i)

        if prompt_tokens > max_prompt_tokens:
            skipped_too_long.append({
                "index": i,
                "ID": case_id,
                "prompt_tokens": prompt_tokens,
                "max_prompt_tokens": max_prompt_tokens,
                "eval_context_limit": args.eval_context_limit,
                "max_new_tokens": args.max_new_tokens,
            })
            continue

        examples.append({
            "index": i,
            "ID": get_case_id(ex, i),
            "Instruction": ex.get("instruction"),
            "Input": ex.get("input"),
            "Gold Text": ex.get("output") or "",
            "Gold Object": safe_json_loads(ex.get("output") or ""),
            "Prompt Tokens": prompt_tokens,
        })

        # Both modes receive the same already rendered prompt. The local prompt
        # also remains the source of context-length accounting.
        request_items.append({"prompt": prompt, "user_text": user_text})

    generation_results = [None] * len(request_items)
    generation_errors = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                call_vllm,
                client=client,
                model_name=args.model_name,
                request_item=request_item,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                presence_penalty=args.presence_penalty,
                frequency_penalty=args.frequency_penalty,
                repetition_penalty=args.repetition_penalty,
                seed=args.seed,
                decoding_mode=args.decoding_mode,
                guided_grammar=guided_grammar,
                guided_decoding_backend=args.guided_decoding_backend,
                completion_add_special_tokens=args.completion_add_special_tokens,
                dynamic_processor_config=dynamic_processor_config,
                dynamic_trace=args.dynamic_trace,
            ): i
            for i, request_item in enumerate(request_items)
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="vLLM inference"):
            i = futures[fut]

            try:
                generation_results[i] = fut.result()

            except Exception as e:
                generation_results[i] = {
                    "text": "",
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "generated_token_ids": None,
                    "finish_reason": "error",
                }
                error_record = {
                    "ID": examples[i]["ID"],
                    "error": str(e),
                }
                generation_errors.append(error_record)
                print(
                    f"[ERROR] ID={examples[i]['ID']} generation failed: {e}",
                    file=sys.stderr,
                )

    if generation_errors:
        preview = "; ".join(
            f"ID={item['ID']}: {item['error']}"
            for item in generation_errors[:5]
        )
        if len(generation_errors) > 5:
            preview += f"; ... {len(generation_errors) - 5} more"
        raise RuntimeError(
            "Aborting evaluation because "
            f"{len(generation_errors)}/{len(request_items)} vLLM request(s) "
            f"failed. No metrics or output files were written. {preview}"
        )


    all_cases = []

    trace_runtime = None
    if dynamic_constrained and args.dynamic_trace:
        trace_runtime = get_cached_runtime(
            args.tokenizer_dir,
            **dynamic_runtime_config,
        )

    all_prompt_tokens = []
    all_completion_tokens = []
    all_total_tokens = []

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    for i, ex_info in enumerate(tqdm(examples, desc="Scoring")):
        generation = generation_results[i] or {}

        pred_text = generation.get("text") or ""
        pred_obj_dense = safe_json_loads(pred_text)
        pred_obj = normalize_prediction(
            pred_obj_dense,
            template_style=(
                args.dynamic_template_style if dynamic_constrained else "canonical"
            ),
        )

        dynamic_validation = None
        dynamic_trace_payload = None
        if dynamic_constrained:
            dynamic_validation = validate_dense_prediction(
                pred_obj_dense,
                template_style=args.dynamic_template_style,
                raw_text=pred_text,
            )
        if trace_runtime is not None:
            generated_token_ids = list(generation.get("generated_token_ids") or [])
            if (
                generated_token_ids
                and generated_token_ids[-1] == trace_runtime.eos_token_id
                and not trace_runtime.compiled_program_ends_with_eos(
                    dynamic_validation.get("number_of_nodules")
                    if dynamic_validation
                    and dynamic_validation.get("number_of_nodules") is not None
                    else 1
                )
            ):
                generated_token_ids.pop()
            try:
                dynamic_trace_payload = TemplateStateMachine(trace_runtime).trace(
                    generated_token_ids
                )
                print(f"\n[TRACE ID={ex_info['ID']}]\n{format_trace(dynamic_trace_payload)}")
            except Exception as exc:
                dynamic_trace_payload = {
                    "error": str(exc),
                    "generated_token_ids": generated_token_ids,
                }

        server_prompt_tokens = generation.get("prompt_tokens")
        completion_tokens = generation.get("completion_tokens")
        total_tokens = generation.get("total_tokens")

        if server_prompt_tokens is not None:
            all_prompt_tokens.append(server_prompt_tokens)

        if completion_tokens is not None:
            all_completion_tokens.append(completion_tokens)

        if total_tokens is not None:
            all_total_tokens.append(total_tokens)

        metrics = score_case(
            gold_obj=normalize_prediction(ex_info["Gold Object"]),
            pred_obj=pred_obj,
            min_similarity=args.min_similarity,
        )

        total_tp += metrics["TP"]
        total_fp += metrics["FP"]
        total_fn += metrics["FN"]


        case_record = {
            "ID": ex_info["ID"],

            "Token Usage": {
                "Prompt Tokens": server_prompt_tokens,
                "Completion Tokens": completion_tokens,
                "Total Tokens": total_tokens,
                "Locally Counted Prompt Tokens": ex_info["Prompt Tokens"],
            },

            "Input": ex_info["Input"],
            "Ground Truth": ex_info["Gold Object"],
            "Prediction": pred_obj,
            "Parsed Prediction Dense": pred_obj_dense,
            "Raw Ground Truth": ex_info["Gold Text"],
            "Raw Prediction": pred_text,
            "Eval Metrics": metrics,
            "Generation Finish Reason": generation.get("finish_reason"),
        }
        if dynamic_constrained:
            case_record["Dynamic Constraint Validation"] = dynamic_validation
        if args.dynamic_trace:
            case_record["Generated Token IDs"] = generation.get("generated_token_ids")
            case_record["Dynamic Constraint Trace"] = dynamic_trace_payload

        all_cases.append(case_record)

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn) if 2 * total_tp + total_fp + total_fn > 0 else 0.0
    iou = total_tp / (total_tp + total_fp + total_fn) if total_tp + total_fp + total_fn > 0 else 0.0

    summary = {
        "Test File": str(Path(args.test_file).expanduser().resolve()),
        "Test File Basename": Path(args.test_file).name,
        "Instruction Provenance": summarize_instructions(
            item.get("Instruction") or "" for item in examples
        ),
        "Original Dataset Cases": len(ds),
        "Total Evaluated Cases": len(all_cases),
        "Skipped Too Long": len(skipped_too_long),
        "Eval Context Limit": args.eval_context_limit,
        "Max New Tokens": args.max_new_tokens,
        "Max Prompt Tokens": max_prompt_tokens,
        "Decoding Mode": args.decoding_mode,
        "Completion Add Special Tokens": (
            args.completion_add_special_tokens
            if args.completion_add_special_tokens is not None
            else (False if constrained else None)
        ),
        "Request Sampling Configuration": (
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "min_p": args.min_p,
                "presence_penalty": args.presence_penalty,
                "frequency_penalty": args.frequency_penalty,
                "repetition_penalty": args.repetition_penalty,
                "seed": args.seed,
            }
            if not dynamic_constrained
            else {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
                "seed": None,
                "note": "legacy sampling is applied inside the custom processor",
            }
        ),
        "Schema File": args.schema_file if grammar_constrained else None,
        "Template File": args.template_file if dynamic_constrained else None,
        "JSON Begin Tag": (
            args.json_begin_tag
            if grammar_constrained or (
                dynamic_constrained and args.dynamic_include_json_tags
            )
            else None
        ),
        "JSON End Tag": (
            args.json_end_tag
            if grammar_constrained or (
                dynamic_constrained and args.dynamic_include_json_tags
            )
            else None
        ),
        "Guided Decoding Backend": (
            args.guided_decoding_backend if grammar_constrained else None
        ),
        "JSON Indent": args.json_indent if constrained else None,
        "Dynamic Template Configuration": (
            dynamic_processor_config if dynamic_constrained else None
        ),
        "Dynamic Template Processor": (
            PROCESSOR_QUALNAME if dynamic_constrained else None
        ),
        "Authentic Dynamic Constraint": dynamic_constrained,
        "Legacy Fidelity Note": (
            "literal forcing; per-slot candidate restriction; prefix filtering; "
            ">=50 token vocabularies with finite typed-prefix validation; sampling "
            "filters applied inside the legal token set; count-dependent template "
            "expansion; original unbounded flattened-language defect repaired"
            if dynamic_constrained and args.dynamic_legacy_compat
            else (
                "full-candidate trie mode"
                if dynamic_constrained else None
            )
        ),
        "Flattened Candidate Safety": (
            "typed finite prefixes; terminal null; semantic post-validation"
            if dynamic_constrained and args.dynamic_legacy_compat
            else None
        ),
        "Prediction Normalization": (
            "canonical key mapping when needed; recursive null pruning; deterministic "
            "integer/float coercion; count 0 plus empty/null Nodules normalized to sparse form"
        ),
        "Date Scoring Normalization": DATE_SCORING_POLICY,

        "Token Usage": {
            "Prompt Tokens": summarize_token_counts(all_prompt_tokens),
            "Completion Tokens": summarize_token_counts(all_completion_tokens),
            "Total Tokens": summarize_token_counts(all_total_tokens),
        },

        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": iou,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, indent=2, ensure_ascii=False)

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nMICRO AVERAGE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
