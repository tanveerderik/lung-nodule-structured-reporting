import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile per-model evaluation summaries side by side."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Directory containing one subdirectory per model.",
    )
    parser.add_argument(
        "--summary_filename",
        default="sft_eval_summary.json",
        help="Summary JSON filename inside each model directory.",
    )
    parser.add_argument(
        "--output_prefix",
        default="sft_eval_summary_side_by_side",
        help="Output filename prefix, without extension.",
    )
    return parser.parse_args()


def flatten(d, prefix=""):
    out = {}

    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k

        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v

    return out


def main():
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()

    pattern = f"*/{args.summary_filename}"

    rows = []

    for path in sorted(results_dir.glob(pattern)):
        model = path.parent.name

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        row = {
            "model": model,
            "summary_file": str(path),
        }
        row.update(flatten(data))
        rows.append(row)

    if not rows:
        raise SystemExit(
            f"No files found under: {results_dir}/{pattern}"
        )

    df = pd.DataFrame(rows)

    main_cols = [
        "model",
        "Precision",
        "Recall",
        "F1",
        "IoU",
        "TP",
        "FP",
        "FN",
    ]

    token_cols = sorted(
        column
        for column in df.columns
        if "token" in column.lower()
    )

    display_cols = [
        column
        for column in main_cols
        if column in df.columns
    ]

    display_cols.extend(
        column
        for column in token_cols
        if column not in display_cols
    )

    print("\nIncluded token columns:")
    for column in token_cols:
        print(" -", column)

    df = df[display_cols]

    if "F1" in df.columns:
        df = df.sort_values("F1", ascending=False)

    df = df.reset_index(drop=True)

    out_csv = results_dir / f"{args.output_prefix}.csv"
    out_md = results_dir / f"{args.output_prefix}.md"

    df.to_csv(out_csv, index=False)
    df.to_markdown(out_md, index=False)

    print("\nSaved:")
    print(out_csv)
    print(out_md)

    print("\nModels included:")
    for model in sorted(df["model"].unique()):
        print(" -", model)

    print("\nSide-by-side summary:\n")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
