#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = [
    "gold_appearances",
    "TP",
    "FP",
    "FN",
    "precision",
    "recall",
    "f1",
]


def load_featurewise_json(path: Path) -> pd.DataFrame:
    """Load and validate a featurewise evaluation JSON file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")

    df = pd.DataFrame(data)

    required = {"feature", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required fields: {sorted(missing)}"
        )

    if df["feature"].duplicated().any():
        duplicates = df.loc[
            df["feature"].duplicated(keep=False), "feature"
        ].tolist()
        raise ValueError(
            f"{path} contains duplicate features: {duplicates}"
        )

    for column in METRICS:
        df[column] = pd.to_numeric(df[column], errors="raise")

    return df


def safe_micro_metrics(tp: int, fp: int, fn: int) -> dict:
    precision_denominator = tp + fp
    recall_denominator = tp + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0
        else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def weighted_average(
    df: pd.DataFrame,
    value_column: str,
    weight_column: str,
) -> float:
    weights = df[weight_column].astype(float)
    total_weight = weights.sum()

    if total_weight == 0:
        return 0.0

    return float(
        (df[value_column].astype(float) * weights).sum()
        / total_weight
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare featurewise precision, recall, and F1 between "
            "an SFT model and a GRPO model."
        )
    )

    parser.add_argument(
        "--sft_json",
        required=True,
        type=Path,
        help="Featurewise JSON generated for the SFT model.",
    )
    parser.add_argument(
        "--grpo_json",
        required=True,
        type=Path,
        help="Featurewise JSON generated for the GRPO model.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("featurewise_f1_comparison.csv"),
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("featurewise_f1_comparison.json"),
    )
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=Path("featurewise_f1_comparison_summary.json"),
    )
    parser.add_argument(
        "--sort_by",
        choices=[
            "feature",
            "delta_f1",
            "absolute_delta_f1",
            "sft_f1",
            "grpo_f1",
        ],
        default="delta_f1",
        help="How to sort the output comparison table.",
    )

    args = parser.parse_args()

    sft = load_featurewise_json(args.sft_json)
    grpo = load_featurewise_json(args.grpo_json)

    sft_features = set(sft["feature"])
    grpo_features = set(grpo["feature"])

    missing_in_grpo = sorted(sft_features - grpo_features)
    missing_in_sft = sorted(grpo_features - sft_features)

    if missing_in_grpo or missing_in_sft:
        raise ValueError(
            "Feature sets do not match.\n"
            f"Missing in GRPO: {missing_in_grpo}\n"
            f"Missing in SFT: {missing_in_sft}"
        )

    comparison = sft.merge(
        grpo,
        on="feature",
        how="inner",
        suffixes=("_sft", "_grpo"),
        validate="one_to_one",
    )

    # Verify that both evaluations used the same reference feature counts.
    comparison["gold_appearances_match"] = (
        comparison["gold_appearances_sft"]
        == comparison["gold_appearances_grpo"]
    )

    for metric in ["TP", "FP", "FN", "precision", "recall", "f1"]:
        comparison[f"delta_{metric}"] = (
            comparison[f"{metric}_grpo"]
            - comparison[f"{metric}_sft"]
        )

    comparison["absolute_delta_f1"] = comparison["delta_f1"].abs()

    tolerance = 1e-12
    comparison["f1_result"] = "unchanged"
    comparison.loc[
        comparison["delta_f1"] > tolerance,
        "f1_result",
    ] = "improved"
    comparison.loc[
        comparison["delta_f1"] < -tolerance,
        "f1_result",
    ] = "worsened"

    # Percentage-point columns are easier to read in tables.
    comparison["delta_precision_pp"] = (
        100.0 * comparison["delta_precision"]
    )
    comparison["delta_recall_pp"] = (
        100.0 * comparison["delta_recall"]
    )
    comparison["delta_f1_pp"] = 100.0 * comparison["delta_f1"]

    # Relative F1 change, expressed as a percentage.
    comparison["relative_f1_change_percent"] = (
        100.0
        * comparison["delta_f1"]
        / comparison["f1_sft"].replace(0, pd.NA)
    )

    output_columns = [
        "feature",
        "gold_appearances_sft",
        "gold_appearances_grpo",
        "gold_appearances_match",
        "TP_sft",
        "TP_grpo",
        "delta_TP",
        "FP_sft",
        "FP_grpo",
        "delta_FP",
        "FN_sft",
        "FN_grpo",
        "delta_FN",
        "precision_sft",
        "precision_grpo",
        "delta_precision",
        "delta_precision_pp",
        "recall_sft",
        "recall_grpo",
        "delta_recall",
        "delta_recall_pp",
        "f1_sft",
        "f1_grpo",
        "delta_f1",
        "delta_f1_pp",
        "relative_f1_change_percent",
        "absolute_delta_f1",
        "f1_result",
    ]

    comparison = comparison[output_columns]

    if args.sort_by == "feature":
        comparison = comparison.sort_values(
            "feature",
            ascending=True,
        )
    elif args.sort_by == "delta_f1":
        comparison = comparison.sort_values(
            ["delta_f1", "feature"],
            ascending=[False, True],
        )
    elif args.sort_by == "absolute_delta_f1":
        comparison = comparison.sort_values(
            ["absolute_delta_f1", "feature"],
            ascending=[False, True],
        )
    elif args.sort_by == "sft_f1":
        comparison = comparison.sort_values(
            ["f1_sft", "feature"],
            ascending=[False, True],
        )
    elif args.sort_by == "grpo_f1":
        comparison = comparison.sort_values(
            ["f1_grpo", "feature"],
            ascending=[False, True],
        )

    comparison = comparison.reset_index(drop=True)

    # Macro averages: every feature receives equal weight.
    macro_sft = {
        metric: float(comparison[f"{metric}_sft"].mean())
        for metric in ["precision", "recall", "f1"]
    }
    macro_grpo = {
        metric: float(comparison[f"{metric}_grpo"].mean())
        for metric in ["precision", "recall", "f1"]
    }

    # Weighted averages: weighted by the gold frequency of each feature.
    weighted_sft = {
        metric: weighted_average(
            comparison,
            f"{metric}_sft",
            "gold_appearances_sft",
        )
        for metric in ["precision", "recall", "f1"]
    }
    weighted_grpo = {
        metric: weighted_average(
            comparison,
            f"{metric}_grpo",
            "gold_appearances_grpo",
        )
        for metric in ["precision", "recall", "f1"]
    }

    # Aggregate count-based micro metrics.
    sft_tp = int(comparison["TP_sft"].sum())
    sft_fp = int(comparison["FP_sft"].sum())
    sft_fn = int(comparison["FN_sft"].sum())

    grpo_tp = int(comparison["TP_grpo"].sum())
    grpo_fp = int(comparison["FP_grpo"].sum())
    grpo_fn = int(comparison["FN_grpo"].sum())

    micro_sft = safe_micro_metrics(sft_tp, sft_fp, sft_fn)
    micro_grpo = safe_micro_metrics(grpo_tp, grpo_fp, grpo_fn)

    improved = comparison[
        comparison["f1_result"] == "improved"
    ]
    worsened = comparison[
        comparison["f1_result"] == "worsened"
    ]
    unchanged = comparison[
        comparison["f1_result"] == "unchanged"
    ]

    summary = {
        "sft_file": str(args.sft_json),
        "grpo_file": str(args.grpo_json),
        "number_of_features": int(len(comparison)),
        "all_gold_appearance_counts_match": bool(
            comparison["gold_appearances_match"].all()
        ),
        "feature_counts": {
            "improved": int(len(improved)),
            "worsened": int(len(worsened)),
            "unchanged": int(len(unchanged)),
        },
        "macro_average": {
            "sft": macro_sft,
            "grpo": macro_grpo,
            "delta": {
                metric: macro_grpo[metric] - macro_sft[metric]
                for metric in ["precision", "recall", "f1"]
            },
        },
        "gold_frequency_weighted_average": {
            "sft": weighted_sft,
            "grpo": weighted_grpo,
            "delta": {
                metric: weighted_grpo[metric] - weighted_sft[metric]
                for metric in ["precision", "recall", "f1"]
            },
        },
        "micro_average_from_aggregated_counts": {
            "sft": {
                "TP": sft_tp,
                "FP": sft_fp,
                "FN": sft_fn,
                **micro_sft,
            },
            "grpo": {
                "TP": grpo_tp,
                "FP": grpo_fp,
                "FN": grpo_fn,
                **micro_grpo,
            },
            "delta": {
                "TP": grpo_tp - sft_tp,
                "FP": grpo_fp - sft_fp,
                "FN": grpo_fn - sft_fn,
                "precision": (
                    micro_grpo["precision"]
                    - micro_sft["precision"]
                ),
                "recall": (
                    micro_grpo["recall"]
                    - micro_sft["recall"]
                ),
                "f1": micro_grpo["f1"] - micro_sft["f1"],
            },
        },
        "largest_improvements": (
            comparison.sort_values(
                "delta_f1",
                ascending=False,
            )
            .head(10)[
                [
                    "feature",
                    "f1_sft",
                    "f1_grpo",
                    "delta_f1",
                    "delta_f1_pp",
                ]
            ]
            .to_dict(orient="records")
        ),
        "largest_regressions": (
            comparison.sort_values(
                "delta_f1",
                ascending=True,
            )
            .head(10)[
                [
                    "feature",
                    "f1_sft",
                    "f1_grpo",
                    "delta_f1",
                    "delta_f1_pp",
                ]
            ]
            .to_dict(orient="records")
        ),
    }

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.summary_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        args.output_csv,
        index=False,
        float_format="%.10f",
    )

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(
            comparison.to_dict(orient="records"),
            f,
            indent=2,
            allow_nan=False,
        )

    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            allow_nan=False,
        )

    print("=" * 72)
    print("FEATUREWISE F1 COMPARISON: SFT vs GRPO")
    print("=" * 72)
    print(f"Features: {len(comparison)}")
    print(f"Improved: {len(improved)}")
    print(f"Worsened: {len(worsened)}")
    print(f"Unchanged: {len(unchanged)}")
    print()
    print(
        f"Macro F1 — SFT:  {macro_sft['f1']:.6f}"
    )
    print(
        f"Macro F1 — GRPO: {macro_grpo['f1']:.6f}"
    )
    print(
        "Macro F1 delta:  "
        f"{macro_grpo['f1'] - macro_sft['f1']:+.6f}"
    )
    print()
    print("Largest F1 improvements:")
    for row in (
        comparison.sort_values("delta_f1", ascending=False)
        .head(5)
        .itertuples()
    ):
        print(
            f"  {row.feature}: "
            f"{row.f1_sft:.6f} -> {row.f1_grpo:.6f} "
            f"({row.delta_f1:+.6f})"
        )

    print()
    print("Largest F1 regressions:")
    for row in (
        comparison.sort_values("delta_f1", ascending=True)
        .head(5)
        .itertuples()
    ):
        print(
            f"  {row.feature}: "
            f"{row.f1_sft:.6f} -> {row.f1_grpo:.6f} "
            f"({row.delta_f1:+.6f})"
        )

    print()
    print(f"CSV:     {args.output_csv}")
    print(f"JSON:    {args.output_json}")
    print(f"Summary: {args.summary_json}")


if __name__ == "__main__":
    main()
