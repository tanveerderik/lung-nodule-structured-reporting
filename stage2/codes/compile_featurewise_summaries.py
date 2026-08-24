import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results"

METRICS = [
    "f1",
    "precision",
    "recall",
    "TP",
    "FP",
    "FN",
    "gold_appearances",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile featurewise evaluation results across models."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Directory containing one subdirectory per model.",
    )
    parser.add_argument(
        "--featurewise_filename",
        default="featurewise_f1.json",
        help="Featurewise JSON filename inside each model directory.",
    )
    parser.add_argument(
        "--output_prefix",
        default="featurewise_f1_side_by_side",
        help="Output filename prefix, without extension.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()

    pattern = f"*/{args.featurewise_filename}"
    rows = []

    for path in sorted(results_dir.glob(pattern)):
        model = path.parent.name

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            row = {
                "model": model,
                "feature": item["feature"],
                "source_file": str(path),
            }

            for metric in METRICS:
                row[metric] = item.get(metric)

            rows.append(row)

    if not rows:
        raise SystemExit(
            f"No files found under: {results_dir}/{pattern}"
        )

    df = pd.DataFrame(rows)

    out_xlsx = results_dir / f"{args.output_prefix}.xlsx"
    out_long_csv = results_dir / f"{args.output_prefix}_long.csv"

    df.to_csv(out_long_csv, index=False)

    f1_table = df.pivot_table(
        index="feature",
        columns="model",
        values="f1",
        aggfunc="first",
    )

    feature_order = (
        f1_table
        .assign(_mean=lambda x: x.mean(axis=1))
        .sort_values("_mean", ascending=False)
        .index
    )

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.sort_values(["feature", "model"]).to_excel(
            writer,
            sheet_name="long",
            index=False,
        )

        for metric in METRICS:
            wide = df.pivot_table(
                index="feature",
                columns="model",
                values=metric,
                aggfunc="first",
            )

            wide = wide.reindex(feature_order)

            wide.reset_index().to_excel(
                writer,
                sheet_name=metric[:31],
                index=False,
            )

    print("\nSaved:")
    print(out_xlsx)
    print(out_long_csv)

    print("\nModels included:")
    for model in sorted(df["model"].unique()):
        print(" -", model)


if __name__ == "__main__":
    main()
