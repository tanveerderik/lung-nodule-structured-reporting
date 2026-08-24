#!/usr/bin/env python3
"""Paired report-level bootstrap for the Stage 2 exact-match evaluation.

The script deliberately reuses the archived featurewise scorer.  It first
materializes additive TP/FP/FN contributions for every report and schema field,
then resamples reports with replacement.  No model inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = [
    "llama3_2_1B",
    "gemma3_4B",
    "mistral_7B",
    "qwen2_5_7B",
    "llama3_1_8B",
    "llama3_1_70B",
]

DISPLAY = {
    "llama3_2_1B": "Llama-3.2-1B",
    "gemma3_4B": "Gemma-3-4B",
    "mistral_7B": "Mistral-7B",
    "qwen2_5_7B": "Qwen2.5-7B",
    "llama3_1_8B": "Llama-3.1-8B",
    "llama3_1_70B": "Llama-3.1-70B",
}

CONFIGS = {
    "base_sparse": (
        "Unadapted, sparse prompt",
        "results/{model}/base_eval_cases_controlled_ordinary.json",
        "results/{model}/base_featurewise_f1_controlled_ordinary.csv",
    ),
    "base_dense": (
        "Unadapted, dense prompt",
        "results_dense_prompt/{model}/base_eval_cases_controlled_ordinary.json",
        "results_dense_prompt/{model}/base_featurewise_f1_controlled_ordinary.csv",
    ),
    "base_dense_dynamic": (
        "Unadapted, dense prompt + dynamic template",
        "results_dense_prompt/{model}/base_eval_cases_controlled_dynamic_template.json",
        "results_dense_prompt/{model}/base_featurewise_f1_controlled_dynamic_template.csv",
    ),
    "sft": (
        "SFT, short prompt",
        "results/{model}/sft_eval_cases_controlled_ordinary.json",
        "results/{model}/sft_featurewise_f1_controlled_ordinary.csv",
    ),
    "sft_grammar": (
        "SFT + sparse grammar",
        "results/{model}/sft_eval_cases_controlled_xgrammar.json",
        "results/{model}/sft_featurewise_f1_controlled_xgrammar.csv",
    ),
}

CONTRASTS = {
    "sft_minus_base_sparse": ("SFT - unadapted sparse", "sft", "base_sparse"),
    "dense_minus_sparse": ("Dense prompt - sparse prompt", "base_dense", "base_sparse"),
    "dynamic_minus_dense": (
        "Dynamic template - dense prompt",
        "base_dense_dynamic",
        "base_dense",
    ),
    "dynamic_minus_sparse": (
        "Dense + dynamic template - sparse prompt",
        "base_dense_dynamic",
        "base_sparse",
    ),
    "grammar_minus_sft": ("Sparse grammar - SFT", "sft_grammar", "sft"),
}

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "grey": "#7F7F7F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--scorer-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_822)
    return parser.parse_args()


def prepare_schema(fe, schema_path: Path):
    with schema_path.open("r", encoding="utf-8") as stream:
        schema = fe.unwrap_lungs_pleura(json.load(stream))
    if isinstance(schema, dict) and "Nodule Findings" in schema:
        schema = schema["Nodule Findings"]

    leaves = fe.schema_leaves(schema)
    list_paths: dict[str, list[str]] = {}

    def find_lists(value, prefix=""):
        value = fe.canonicalize(value)
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else key
                find_lists(child, path)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            child_fields = [
                key
                for key, child in fe.canonicalize(value[0]).items()
                if isinstance(child, dict) and "data_type" in child
            ]
            list_paths[prefix] = child_fields

    find_lists(schema)
    features = set(leaves)
    for path, child_fields in list_paths.items():
        features.update(f"{path}.{child}" for child in child_fields)
    return leaves, list_paths, sorted(features)


def normalized_gold(fe, case):
    source = case.get("Ground Truth") if "Ground Truth" in case else fe.safe_load(case.get("Raw Ground Truth"))
    return fe.normalize_prediction(fe.unwrap_lungs_pleura(source))


def score_case_featurewise(fe, case, leaves, list_paths, features):
    stats = defaultdict(lambda: {"gold_appearances": 0, "TP": 0, "FP": 0, "FN": 0})
    for feature in features:
        stats[feature]

    gold_source = case.get("Ground Truth") if "Ground Truth" in case else fe.safe_load(case.get("Raw Ground Truth"))
    pred_source = case.get("Prediction") if "Prediction" in case else fe.safe_load(case.get("Raw Prediction"))
    gold = fe.normalize_prediction(fe.unwrap_lungs_pleura(gold_source))
    pred = fe.normalize_prediction(fe.unwrap_lungs_pleura(pred_source))
    handled = set()

    for list_path, child_fields in list_paths.items():
        gold_list = fe.get_path(gold, list_path)
        pred_list = fe.get_path(pred, list_path)
        gold_list = gold_list if isinstance(gold_list, list) else []
        pred_list = pred_list if isinstance(pred_list, list) else []
        matches, unmatched_gold, unmatched_pred = fe.match_records(
            gold_list, pred_list, min_sim=0.20
        )
        for gold_index, pred_index in matches:
            fe.score_dict_leaves(
                stats,
                list_path,
                gold_list[gold_index],
                pred_list[pred_index],
                child_fields,
            )
        for gold_index in unmatched_gold:
            fe.score_dict_leaves(stats, list_path, gold_list[gold_index], {}, child_fields)
        for pred_index in unmatched_pred:
            fe.score_dict_leaves(stats, list_path, {}, pred_list[pred_index], child_fields)
        handled.update(f"{list_path}.{child}" for child in child_fields)

    for feature in leaves:
        if feature not in handled:
            fe.score_scalar(stats, feature, fe.get_path(gold, feature), fe.get_path(pred, feature))

    return np.asarray(
        [
            [
                stats[feature]["gold_appearances"],
                stats[feature]["TP"],
                stats[feature]["FP"],
                stats[feature]["FN"],
            ]
            for feature in features
        ],
        dtype=float,
    )


def f1_from_totals(totals: np.ndarray) -> np.ndarray:
    tp = totals[..., 0]
    fp = totals[..., 1]
    fn = totals[..., 2]
    denominator = 2 * tp + fp + fn
    return np.divide(
        2 * tp,
        denominator,
        out=np.full_like(denominator, np.nan, dtype=float),
        where=denominator > 0,
    )


def gold_support_weighted_f1(feature_totals: np.ndarray) -> np.ndarray:
    """Gold-support-weighted mean of feature-level exact-match F1.

    ``feature_totals`` must have a final dimension of four columns in the
    order ``[gold_appearances, TP, FP, FN]`` and a penultimate feature
    dimension.  The weight for feature j is its number of non-null gold
    appearances in the evaluated sample.  Features with zero gold support in
    a bootstrap replicate receive zero weight in that replicate.

    This is intentionally distinct from micro-F1: feature-level F1 is
    calculated first, then averaged using gold-reference support only.
    """
    support = feature_totals[..., 0]
    feature_f1 = f1_from_totals(feature_totals[..., 1:4])
    weighted_terms = np.where(
        support > 0,
        support * np.nan_to_num(feature_f1, nan=0.0),
        0.0,
    )
    numerator = weighted_terms.sum(axis=-1)
    denominator = support.sum(axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(denominator, np.nan, dtype=float),
        where=denominator > 0,
    )


def percentile_ci(values: np.ndarray) -> tuple[float, float, int]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return math.nan, math.nan, 0
    low, high = np.percentile(valid, [2.5, 97.5])
    return float(low), float(high), int(valid.size)


def bootstrap_weights(rng: np.random.Generator, iterations: int, n: int) -> np.ndarray:
    draws = rng.integers(0, n, size=(iterations, n), dtype=np.int32)
    weights = np.zeros((iterations, n), dtype=np.int16)
    rows = np.repeat(np.arange(iterations, dtype=np.int32), n)
    np.add.at(weights, (rows, draws.ravel()), 1)
    return weights


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No rows generated for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tex_ci(point, low, high) -> str:
    return f"{point:.3f} ({low:.3f}, {high:.3f})"


def tex_delta(point, low, high) -> str:
    return f"{point * 100:+.2f} ({low * 100:+.2f}, {high * 100:+.2f})"


def short_feature(name: str) -> str:
    return name.removeprefix("Nodules.").replace("_", " ")


def save_latex_fragments(output_dir, overall, deltas, featurewise, feature_deltas, sensitivity, token_csv):
    overall_by = {(row["model"], row["configuration"]): row for row in overall}
    delta_by = {(row["model"], row["contrast"]): row for row in deltas}
    token_df = pd.read_csv(token_csv)
    token_by = {
        (row["model"], row["regime"]): row
        for row in token_df.to_dict(orient="records")
    }

    table8 = []
    table9 = []
    table10 = []
    for model in MODELS:
        base = overall_by[(model, "base_sparse")]
        dense = overall_by[(model, "base_dense")]
        dynamic = overall_by[(model, "base_dense_dynamic")]
        sft = overall_by[(model, "sft")]
        grammar = overall_by[(model, "sft_grammar")]
        sft_delta = delta_by[(model, "sft_minus_base_sparse")]
        constraint_delta = delta_by[(model, "dynamic_minus_dense")]
        grammar_delta = delta_by[(model, "grammar_minus_sft")]
        base_tokens = token_by[(model, "base")]
        sft_tokens = token_by[(model, "sft")]
        reduction = 1 - sft_tokens["Token Usage.Total Tokens.Mean"] / base_tokens["Token Usage.Total Tokens.Mean"]

        table8.append(
            f"{DISPLAY[model]}{'*' if model == 'llama3_1_70B' else ''} & "
            f"{tex_ci(base['f1'], base['ci_low'], base['ci_high'])} & "
            f"{tex_ci(sft['f1'], sft['ci_low'], sft['ci_high'])} & "
            f"{tex_delta(sft_delta['delta'], sft_delta['ci_low'], sft_delta['ci_high'])} & "
            f"{base_tokens['Token Usage.Total Tokens.Mean']:,.0f} & "
            f"{sft_tokens['Token Usage.Total Tokens.Mean']:,.0f} & {reduction * 100:.1f}\\% \\\\"
        )
        table9.append(
            f"{DISPLAY[model]}{'*' if model == 'llama3_1_70B' else ''} & "
            f"{tex_ci(base['f1'], base['ci_low'], base['ci_high'])} & "
            f"{tex_ci(dense['f1'], dense['ci_low'], dense['ci_high'])} & "
            f"{tex_ci(dynamic['f1'], dynamic['ci_low'], dynamic['ci_high'])} & "
            f"{tex_delta(constraint_delta['delta'], constraint_delta['ci_low'], constraint_delta['ci_high'])} \\\\"
        )
        table10.append(
            f"{DISPLAY[model]}{'*' if model == 'llama3_1_70B' else ''} & "
            f"{sft['raw_sparse_validity']:.1f}\\% & {sft['raw_dense_validity']:.1f}\\% & 100.0\\% & "
            f"{sft['ordinary_valid_count']}/250 & 250/250 & "
            f"{tex_ci(sft['f1'], sft['ci_low'], sft['ci_high'])} & "
            f"{tex_ci(grammar['f1'], grammar['ci_low'], grammar['ci_high'])} & "
            f"{tex_delta(grammar_delta['delta'], grammar_delta['ci_low'], grammar_delta['ci_high'])} \\\\"
        )

    (output_dir / "table8_ci_rows.tex").write_text(
        "\n".join(table8) + "\n\\bottomrule\n", encoding="utf-8"
    )
    (output_dir / "table9_ci_rows.tex").write_text(
        "\n".join(table9) + "\n\\bottomrule\n", encoding="utf-8"
    )
    (output_dir / "table10_ci_rows.tex").write_text(
        "\n".join(table10) + "\n\\bottomrule\n", encoding="utf-8"
    )

    qwen_abs = {
        row["feature"]: row
        for row in featurewise
        if row["model"] == "qwen2_5_7B" and row["configuration"] == "sft"
    }
    qwen_delta = {
        row["feature"]: row
        for row in feature_deltas
        if row["model"] == "qwen2_5_7B" and row["contrast"] == "grammar_minus_sft"
    }
    feature_rows = []
    for feature in sorted(qwen_abs, key=lambda item: (-int(qwen_abs[item]["support"]), item)):
        absolute = qwen_abs[feature]
        delta = qwen_delta[feature]
        support = int(absolute["support"])
        escaped = short_feature(feature).replace("%", "\\%").replace("_", "\\_")
        if support >= 20:
            absolute_text = tex_ci(absolute["f1"], absolute["ci_low"], absolute["ci_high"])
            delta_text = tex_delta(delta["delta"], delta["ci_low"], delta["ci_high"])
        else:
            absolute_text = f"{absolute['f1']:.3f} (descriptive)"
            delta_text = f"{delta['delta'] * 100:+.2f} (descriptive)"
        feature_rows.append(f"{escaped} & {support} & {absolute_text} & {delta_text} \\\\")
    (output_dir / "tableS2_featurewise_rows.tex").write_text(
        "\n".join(feature_rows) + "\n", encoding="utf-8"
    )

    sensitivity_rows = []
    for row in sensitivity:
        if row["configuration"] != "sft":
            continue
        sensitivity_rows.append(
            f"{DISPLAY[row['model']]}{'*' if row['model'] == 'llama3_1_70B' else ''} & "
            f"{tex_ci(row['overall_f1'], row['overall_ci_low'], row['overall_ci_high'])} & "
            f"{tex_ci(row['positive_f1'], row['positive_ci_low'], row['positive_ci_high'])} & "
            f"{row['negative_fp_reports']}/142 ({row['negative_fp_rate'] * 100:.1f}\\%; "
            f"95\\% CI: {row['negative_fp_ci_low'] * 100:.1f}, {row['negative_fp_ci_high'] * 100:.1f}\\%) \\\\"
        )
    (output_dir / "tableS3_sensitivity_rows.tex").write_text(
        "\n".join(sensitivity_rows) + "\n\\bottomrule\n", encoding="utf-8"
    )


def make_figures(output_dir, overall, deltas, featurewise, feature_deltas, token_csv):
    overall_by = {(row["model"], row["configuration"]): row for row in overall}
    delta_by = {(row["model"], row["contrast"]): row for row in deltas}
    y = np.arange(len(MODELS))
    labels = [DISPLAY[model] for model in MODELS]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    styles = [
        ("base_sparse", -0.18, "o", COLORS["blue"], "Unadapted, sparse prompt"),
        ("base_dense", 0.0, "s", COLORS["orange"], "Unadapted, dense prompt"),
        (
            "base_dense_dynamic",
            0.18,
            "D",
            COLORS["vermillion"],
            "Unadapted, dense prompt + dynamic template",
        ),
    ]
    for config, offset, marker, color, label in styles:
        points = np.asarray([overall_by[(model, config)]["f1"] for model in MODELS])
        lows = np.asarray([overall_by[(model, config)]["ci_low"] for model in MODELS])
        highs = np.asarray([overall_by[(model, config)]["ci_high"] for model in MODELS])
        ax.errorbar(
            points,
            y + offset,
            xerr=np.vstack([points - lows, highs - points]),
            fmt=marker,
            markersize=7,
            capsize=3,
            elinewidth=1.2,
            color=color,
            ecolor=color,
            label=label,
            zorder=3,
        )
    for index, model in enumerate(MODELS):
        dense_row = overall_by[(model, "base_dense")]
        dynamic_row = overall_by[(model, "base_dense_dynamic")]
        dense = dense_row["f1"]
        dynamic = dynamic_row["f1"]
        delta = delta_by[(model, "dynamic_minus_dense")]["delta"] * 100
        ax.plot([dense, dynamic], [index, index], color="#D0D0D0", linewidth=1.4, zorder=1)
        color = COLORS["vermillion"] if delta >= 0 else COLORS["grey"]
        label_x = max(dense_row["ci_high"], dynamic_row["ci_high"]) + 0.018
        ax.text(label_x, index, f"{delta:+.2f} pp", va="center", color=color)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Exact-match micro-F1")
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "figV2_basematrix_ci.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    token_df = pd.read_csv(token_csv)
    token_by = {
        (row["model"], row["regime"]): row
        for row in token_df.to_dict(orient="records")
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 6.1), gridspec_kw={"wspace": 0.30})
    for index, model in enumerate(MODELS):
        base = overall_by[(model, "base_sparse")]
        sft = overall_by[(model, "sft")]
        delta = delta_by[(model, "sft_minus_base_sparse")]["delta"] * 100
        ax1.plot([base["f1"], sft["f1"]], [index, index], color="#71B783", linewidth=2.3, zorder=1)
        for row, marker, color in [(base, "o", COLORS["blue"]), (sft, "^", COLORS["green"])]:
            ax1.errorbar(
                row["f1"],
                index,
                xerr=[[row["f1"] - row["ci_low"]], [row["ci_high"] - row["f1"]]],
                fmt=marker,
                markersize=7,
                capsize=3,
                elinewidth=1.2,
                color=color,
                ecolor=color,
                zorder=3,
            )
        label_x = max(base["ci_high"], sft["ci_high"]) + 0.020
        ax1.text(label_x, index, f"{delta:+.1f} pp", va="center", color="#4DAF6A" if delta >= 0 else "#D65F5F")
    ax1.set_yticks(y, labels)
    ax1.set_xlim(0, 1.17)
    ax1.set_xlabel("Exact-match micro-F1")
    ax1.grid(axis="x", color="#E0E0E0", linewidth=0.8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title("(a) Unadapted sparse prompt vs SFT")
    ax1.scatter([], [], marker="o", color=COLORS["blue"], label="Unadapted, sparse prompt")
    ax1.scatter([], [], marker="^", color=COLORS["green"], label="SFT, short prompt")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.52, 1.10), ncol=2, frameon=False, fontsize=9)

    height = 0.32
    for index, model in enumerate(MODELS):
        for regime, offset, color in [("base", 0.17, COLORS["blue"]), ("sft", -0.17, COLORS["green"])]:
            row = token_by[(model, regime)]
            prompt = row["Token Usage.Prompt Tokens.Mean"]
            completion = row["Token Usage.Completion Tokens.Mean"]
            ax2.barh(index + offset, prompt, height=height, color=color)
            ax2.barh(
                index + offset,
                completion,
                left=prompt,
                height=height,
                color=color,
                alpha=0.45,
                hatch="//",
                edgecolor="white",
                linewidth=0.4,
            )
            ax2.text(prompt + completion + 75, index + offset, f"{prompt + completion:,.0f}", va="center", fontsize=9)
    ax2.set_yticks(y, [""] * len(y))
    ax2.set_xlim(0, 6900)
    ax2.set_xlabel("Mean tokens per report")
    ax2.grid(axis="x", color="#E0E0E0", linewidth=0.8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("(b) Token expenditure")
    ax2.plot([], [], color="#808080", linewidth=8, label="solid = prompt")
    ax2.plot([], [], color="#BDBDBD", linewidth=8, label="hatched = completion")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False, fontsize=9)
    fig.savefig(output_dir / "figV2_tradeoff_ci.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    abs_rows = [
        row
        for row in featurewise
        if row["model"] == "qwen2_5_7B" and row["configuration"] == "sft" and int(row["support"]) >= 20
    ]
    delta_lookup = {
        row["feature"]: row
        for row in feature_deltas
        if row["model"] == "qwen2_5_7B" and row["contrast"] == "grammar_minus_sft"
    }
    abs_rows.sort(key=lambda row: row["f1"])
    fy = np.arange(len(abs_rows))
    flabels = [f"{short_feature(row['feature'])}  (n={int(row['support'])})" for row in abs_rows]
    points = np.asarray([row["f1"] for row in abs_rows])
    lows = np.asarray([row["ci_low"] for row in abs_rows])
    highs = np.asarray([row["ci_high"] for row in abs_rows])
    drows = [delta_lookup[row["feature"]] for row in abs_rows]
    dpoints = np.asarray([row["delta"] * 100 for row in drows])
    dlows = np.asarray([row["ci_low"] * 100 for row in drows])
    dhighs = np.asarray([row["ci_high"] * 100 for row in drows])

    fig, (fax1, fax2) = plt.subplots(1, 2, figsize=(12.2, 7.0), sharey=True, gridspec_kw={"width_ratios": [1.25, 1]})
    fax1.errorbar(
        points,
        fy,
        xerr=np.vstack([points - lows, highs - points]),
        fmt="o",
        color=COLORS["green"],
        ecolor=COLORS["green"],
        capsize=3,
        markersize=5,
        elinewidth=1.2,
    )
    fax1.set_yticks(fy, flabels)
    fax1.set_xlim(0, 1.02)
    fax1.set_xlabel("Exact-match feature F1")
    fax1.set_title("(a) Qwen2.5-7B after SFT")
    fax1.grid(axis="x", color="#E0E0E0", linewidth=0.8)
    fax1.spines[["top", "right"]].set_visible(False)

    fax2.axvline(0, color="#777777", linewidth=1)
    fax2.errorbar(
        dpoints,
        fy,
        xerr=np.vstack([dpoints - dlows, dhighs - dpoints]),
        fmt="D",
        color=COLORS["vermillion"],
        ecolor=COLORS["vermillion"],
        capsize=3,
        markersize=4.5,
        elinewidth=1.2,
    )
    bound = max(2.0, float(np.nanmax(np.abs(np.concatenate([dlows, dhighs])))) * 1.10)
    fax2.set_xlim(-bound, bound)
    fax2.set_xlabel("Paired grammar effect (percentage points)")
    fax2.set_title("(b) Sparse grammar - SFT")
    fax2.grid(axis="x", color="#E0E0E0", linewidth=0.8)
    fax2.spines[["top", "right", "left"]].set_visible(False)
    fax2.tick_params(axis="y", left=False, labelleft=False)
    fig.tight_layout()
    fig.savefig(output_dir / "figV2_featurewise_bootstrap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.scorer_dir.resolve()))
    import featurewise_eval as fe

    leaves, list_paths, features = prepare_schema(fe, args.schema)
    anchor_ids = None
    anchor_ground_truth = None
    anchor_positive = None
    contributions = {}
    cases_by = {}

    for model in MODELS:
        for config, (_, case_pattern, _) in CONFIGS.items():
            case_path = args.results_root / case_pattern.format(model=model)
            with case_path.open("r", encoding="utf-8") as stream:
                cases = json.load(stream)
            by_id = {case["ID"]: case for case in cases}
            ids = sorted(by_id)
            if len(cases) != 250 or len(ids) != 250:
                raise ValueError(f"{case_path} does not contain 250 unique cases")
            gold = {case_id: canonical_json(normalized_gold(fe, by_id[case_id])) for case_id in ids}
            if anchor_ids is None:
                anchor_ids = ids
                anchor_ground_truth = gold
                anchor_positive = np.asarray(
                    [
                        len((normalized_gold(fe, by_id[case_id]).get("Nodules") or [])) > 0
                        for case_id in ids
                    ],
                    dtype=bool,
                )
            elif ids != anchor_ids or gold != anchor_ground_truth:
                raise ValueError(f"Case IDs or references differ in {case_path}")
            cases_ordered = [by_id[case_id] for case_id in anchor_ids]
            contributions[(model, config)] = np.stack(
                [score_case_featurewise(fe, case, leaves, list_paths, features) for case in cases_ordered],
                axis=0,
            )
            cases_by[(model, config)] = cases_ordered

    support = contributions[(MODELS[0], "sft")][:, :, 0].sum(axis=0)
    supported_mask = support > 0
    if int(supported_mask.sum()) != 25 or int(support[supported_mask].sum()) != 1649:
        raise ValueError(
            f"Expected 25 supported fields and 1,649 instances; got {supported_mask.sum()} and {support[supported_mask].sum()}"
        )

    rng = np.random.default_rng(args.seed)
    weights = bootstrap_weights(rng, args.iterations, 250)
    positive_indices = np.flatnonzero(anchor_positive)
    negative_indices = np.flatnonzero(~anchor_positive)
    positive_weights = bootstrap_weights(rng, args.iterations, len(positive_indices))
    negative_weights = bootstrap_weights(rng, args.iterations, len(negative_indices))

    overall_rows = []
    weighted_rows = []
    feature_rows = []
    sensitivity_rows = []
    overall_boot = {}
    weighted_boot = {}
    feature_boot = {}

    raw_validity_path = args.results_root / "results" / "all_eval_postprocess_manifest.json"
    if not raw_validity_path.exists():
        raise FileNotFoundError(raw_validity_path)
    validity_table = {
        "llama3_2_1B": (5.2, 0.0, 248),
        "gemma3_4B": (21.6, 26.4, 246),
        "mistral_7B": (53.6, 52.8, 247),
        "qwen2_5_7B": (86.0, 84.8, 248),
        "llama3_1_8B": (70.4, 74.8, 248),
        "llama3_1_70B": (92.0, 92.0, 229),
    }

    for model in MODELS:
        for config in CONFIGS:
            array = contributions[(model, config)]
            supported_array = array[:, supported_mask, :]

            # Existing primary metric: exact-match micro-F1 from pooled
            # supported-feature TP/FP/FN counts.
            per_case = supported_array[:, :, 1:4].sum(axis=1)
            point_totals = per_case.sum(axis=0)
            point = float(f1_from_totals(point_totals))
            boot_totals = weights @ per_case
            boot_f1 = f1_from_totals(boot_totals)
            low, high, valid = percentile_ci(boot_f1)
            overall_boot[(model, config)] = boot_f1

            # Sensitivity metric: calculate feature-level F1 first and then
            # average it using each feature's gold-reference appearance count.
            # Bootstrap weights are re-estimated from each resampled cohort.
            point_feature_totals_all = supported_array.sum(axis=0)
            weighted_point = float(gold_support_weighted_f1(point_feature_totals_all))
            boot_feature_totals_all = np.einsum(
                "bi,ifk->bfk", weights, supported_array, optimize=True
            )
            weighted_distribution = gold_support_weighted_f1(boot_feature_totals_all)
            weighted_low, weighted_high, weighted_valid = percentile_ci(weighted_distribution)
            weighted_boot[(model, config)] = weighted_distribution

            _, _, feature_csv_pattern = CONFIGS[config]
            expected_df = pd.read_csv(args.results_root / feature_csv_pattern.format(model=model))
            expected_df = expected_df[expected_df["gold_appearances"] > 0].copy()
            expected = 2 * expected_df.TP.sum() / (
                2 * expected_df.TP.sum() + expected_df.FP.sum() + expected_df.FN.sum()
            )
            expected_weighted = float(
                (expected_df["gold_appearances"] * expected_df["f1"]).sum()
                / expected_df["gold_appearances"].sum()
            )
            if not np.isclose(point, expected, atol=1e-12):
                raise ValueError(f"Exact-match point estimate mismatch for {model}/{config}: {point} vs {expected}")
            if not np.isclose(weighted_point, expected_weighted, atol=1e-12):
                raise ValueError(
                    f"Gold-support-weighted F1 mismatch for {model}/{config}: "
                    f"{weighted_point} vs {expected_weighted}"
                )

            raw_sparse, raw_dense, ordinary_valid = validity_table[model]
            overall_rows.append(
                {
                    "model": model,
                    "model_label": DISPLAY[model],
                    "configuration": config,
                    "configuration_label": CONFIGS[config][0],
                    "f1": point,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_iterations": args.iterations,
                    "valid_bootstrap_replicates": valid,
                    "gold_support_weighted_f1": weighted_point,
                    "gold_support_weighted_ci_low": weighted_low,
                    "gold_support_weighted_ci_high": weighted_high,
                    "gold_support_weighted_valid_bootstrap_replicates": weighted_valid,
                    "TP": point_totals[0],
                    "FP": point_totals[1],
                    "FN": point_totals[2],
                    "raw_sparse_validity": raw_sparse,
                    "raw_dense_validity": raw_dense,
                    "ordinary_valid_count": ordinary_valid,
                }
            )

            weighted_rows.append(
                {
                    "model": model,
                    "model_label": DISPLAY[model],
                    "configuration": config,
                    "configuration_label": CONFIGS[config][0],
                    "gold_support_weighted_f1": weighted_point,
                    "ci_low": weighted_low,
                    "ci_high": weighted_high,
                    "bootstrap_iterations": args.iterations,
                    "valid_bootstrap_replicates": weighted_valid,
                    "supported_feature_count": int(supported_mask.sum()),
                    "gold_reference_appearances": int(support[supported_mask].sum()),
                }
            )

            # Feature-specific CIs retain the complete schema feature index so
            # downstream feature rows remain aligned with ``features``.
            feature_totals = np.einsum(
                "bi,ifk->bfk", weights, array[:, :, 1:4], optimize=True
            )
            feature_f1 = f1_from_totals(feature_totals)
            feature_boot[(model, config)] = feature_f1
            point_feature = f1_from_totals(array[:, :, 1:4].sum(axis=0))
            for feature_index, feature in enumerate(features):
                if not supported_mask[feature_index]:
                    continue
                f_low, f_high, f_valid = percentile_ci(feature_f1[:, feature_index])
                feature_rows.append(
                    {
                        "model": model,
                        "model_label": DISPLAY[model],
                        "configuration": config,
                        "configuration_label": CONFIGS[config][0],
                        "feature": feature,
                        "support": int(support[feature_index]),
                        "f1": float(point_feature[feature_index]),
                        "ci_low": f_low,
                        "ci_high": f_high,
                        "valid_bootstrap_replicates": f_valid,
                    }
                )

            positive_case = per_case[positive_indices]
            positive_f1_point = float(f1_from_totals(positive_case.sum(axis=0)))
            positive_f1_boot = f1_from_totals(positive_weights @ positive_case)
            p_low, p_high, _ = percentile_ci(positive_f1_boot)

            negative_fp_flag = (per_case[negative_indices, 1] > 0).astype(float)
            negative_rate_point = float(negative_fp_flag.mean())
            negative_rate_boot = (negative_weights @ negative_fp_flag) / len(negative_indices)
            n_low, n_high, _ = percentile_ci(negative_rate_boot)
            sensitivity_rows.append(
                {
                    "model": model,
                    "model_label": DISPLAY[model],
                    "configuration": config,
                    "overall_f1": point,
                    "overall_ci_low": low,
                    "overall_ci_high": high,
                    "positive_reports": int(anchor_positive.sum()),
                    "positive_f1": positive_f1_point,
                    "positive_ci_low": p_low,
                    "positive_ci_high": p_high,
                    "negative_reports": int((~anchor_positive).sum()),
                    "negative_fp_reports": int(negative_fp_flag.sum()),
                    "negative_fp_rate": negative_rate_point,
                    "negative_fp_ci_low": n_low,
                    "negative_fp_ci_high": n_high,
                }
            )

    delta_rows = []
    weighted_delta_rows = []
    feature_delta_rows = []
    for model in MODELS:
        for contrast, (label, left, right) in CONTRASTS.items():
            point_left = next(row["f1"] for row in overall_rows if row["model"] == model and row["configuration"] == left)
            point_right = next(row["f1"] for row in overall_rows if row["model"] == model and row["configuration"] == right)
            distribution = overall_boot[(model, left)] - overall_boot[(model, right)]
            low, high, valid = percentile_ci(distribution)
            delta_rows.append(
                {
                    "model": model,
                    "model_label": DISPLAY[model],
                    "contrast": contrast,
                    "contrast_label": label,
                    "delta": point_left - point_right,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_iterations": args.iterations,
                    "valid_bootstrap_replicates": valid,
                }
            )
            weighted_left = next(
                row["gold_support_weighted_f1"]
                for row in weighted_rows
                if row["model"] == model and row["configuration"] == left
            )
            weighted_right = next(
                row["gold_support_weighted_f1"]
                for row in weighted_rows
                if row["model"] == model and row["configuration"] == right
            )
            weighted_difference = weighted_boot[(model, left)] - weighted_boot[(model, right)]
            w_low, w_high, w_valid = percentile_ci(weighted_difference)
            weighted_delta_rows.append(
                {
                    "model": model,
                    "model_label": DISPLAY[model],
                    "contrast": contrast,
                    "contrast_label": label,
                    "delta": weighted_left - weighted_right,
                    "ci_low": w_low,
                    "ci_high": w_high,
                    "bootstrap_iterations": args.iterations,
                    "valid_bootstrap_replicates": w_valid,
                }
            )

            feature_distribution = feature_boot[(model, left)] - feature_boot[(model, right)]
            left_points = {
                row["feature"]: row["f1"]
                for row in feature_rows
                if row["model"] == model and row["configuration"] == left
            }
            right_points = {
                row["feature"]: row["f1"]
                for row in feature_rows
                if row["model"] == model and row["configuration"] == right
            }
            for feature_index, feature in enumerate(features):
                if not supported_mask[feature_index]:
                    continue
                f_low, f_high, f_valid = percentile_ci(feature_distribution[:, feature_index])
                feature_delta_rows.append(
                    {
                        "model": model,
                        "model_label": DISPLAY[model],
                        "contrast": contrast,
                        "contrast_label": label,
                        "feature": feature,
                        "support": int(support[feature_index]),
                        "delta": left_points[feature] - right_points[feature],
                        "ci_low": f_low,
                        "ci_high": f_high,
                        "valid_bootstrap_replicates": f_valid,
                    }
                )

    write_csv(args.output_dir / "bootstrap_overall_f1.csv", overall_rows)
    write_csv(args.output_dir / "bootstrap_paired_deltas.csv", delta_rows)
    write_csv(args.output_dir / "bootstrap_gold_support_weighted_f1.csv", weighted_rows)
    write_csv(args.output_dir / "bootstrap_gold_support_weighted_deltas.csv", weighted_delta_rows)
    write_csv(args.output_dir / "bootstrap_featurewise_f1.csv", feature_rows)
    write_csv(args.output_dir / "bootstrap_featurewise_deltas.csv", feature_delta_rows)
    write_csv(args.output_dir / "bootstrap_case_mix_sensitivity.csv", sensitivity_rows)

    token_csv = args.results_root / "results" / "all_eval_token_usage_side_by_side.csv"
    save_latex_fragments(
        args.output_dir,
        overall_rows,
        delta_rows,
        feature_rows,
        feature_delta_rows,
        sensitivity_rows,
        token_csv,
    )
    make_figures(
        args.output_dir,
        overall_rows,
        delta_rows,
        feature_rows,
        feature_delta_rows,
        token_csv,
    )

    manifest = {
        "method": "paired nonparametric percentile bootstrap at the report level",
        "iterations": args.iterations,
        "seed": args.seed,
        "reports": 250,
        "nodule_positive_reports": int(anchor_positive.sum()),
        "nodule_negative_reports": int((~anchor_positive).sum()),
        "supported_features": int(supported_mask.sum()),
        "reference_feature_instances": int(support[supported_mask].sum()),
        "models": MODELS,
        "configurations": list(CONFIGS),
        "scoring": "exact-match featurewise TP/FP/FN after archived normalization and nodule matching",
        "overall_metrics": {
            "micro_f1": (
                "2*sum(TP)/(2*sum(TP)+sum(FP)+sum(FN)) over the fixed "
                "full-cohort reference-supported feature set"
            ),
            "gold_support_weighted_feature_f1": (
                "sum_j(n_j*F1_j)/sum_j(n_j), where n_j is the number of "
                "non-null gold-reference appearances; n_j is recomputed within "
                "each report-level bootstrap resample and zero-support features "
                "receive zero weight in that replicate"
            ),
        },
        "inference_rerun": False,
    }
    (args.output_dir / "bootstrap_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
