"""
Generates a summary CSV from a detailed evaluation results file.

This script processes a CSV file containing detailed model evaluation results,
typically the output from `evaluate_hf_all_judges.py`. The input file is
expected to have columns for 'model', 'label', and one or more 'judge'
columns prefixed with 'is_refusal_' (e.g., 'is_refusal_gpt-4',
'is_refusal_claude-3-opus').

The script aggregates the data to produce a summary file. For each combination
of model (including its abliterated version), label, and judge, it calculates
the total count of 'refused' and 'not_refused' responses.

The output is a new CSV file (e.g., '..._all_summary.csv') with the following
columns: 'model', 'label', 'refusal_judge', 'refused', 'not_refused', 'total'.
This summary format, which separates results by judge, is specifically designed
to be compatible with the `plotting.ipynb` notebook for visualization.

Usage:
------
This file is meant to be run directly from the command line using `uv run`.
You must provide the path to the input CSV file as an argument.

Example:
# To run this script on the output of evaluate_hf_all_judges.py:
# uv run python src/extract_summary.py path/to/your/model_comparison_..._with_all.csv
"""

import pandas as pd
from pathlib import Path

# Input: the _with_all.csv file
in_path = Path("model_comparison_20250828_150255_with_openai_with_all.csv")

# Output path: match your naming style: *_all_summary.csv
if in_path.name.endswith("_with_all.csv"):
    out_path = in_path.with_name(in_path.name.replace("_with_all.csv", "_all_summary.csv"))
else:
    out_path = in_path.with_name(in_path.stem + "_all_summary.csv")

# Load data
df = pd.read_csv(in_path)

# Sanity check: required base columns
for col in ("model", "label"):
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}' in {in_path}")

# Find all judge columns (must start with 'is_refusal_')
judge_cols = [c for c in df.columns if c.startswith("is_refusal_")]
if not judge_cols:
    raise ValueError("No judge columns found (expected columns starting with 'is_refusal_').")

# Build summary rows per judge, per (model, label)
records = []
for col in judge_cols:
    refusal_judge = col.replace("is_refusal_", "", 1)  # identical judge name as in your reference summary
    # Group by model, label and aggregate refused / not_refused / total
    grp = df.groupby(["model", "label"], dropna=False)[col].agg(
        refused=lambda x: int(pd.Series(x).astype(bool).sum()),
        not_refused=lambda x: int((~pd.Series(x).astype(bool)).sum()),
        total=lambda x: int(pd.Series(x).notna().sum()),
    ).reset_index()

    # Insert the refusal_judge column after model, label
    grp.insert(2, "refusal_judge", refusal_judge)
    records.append(grp)

# Concatenate and enforce exact column order to match the reference summary
summary_df = pd.concat(records, ignore_index=True)[
    ["model", "label", "refusal_judge", "refused", "not_refused", "total"]
]

# Save
summary_df.to_csv(out_path, index=False)
print(f"Wrote: {out_path}")
