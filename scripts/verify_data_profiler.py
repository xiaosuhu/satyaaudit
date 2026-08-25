"""Run DataProfiler against fixtures/data/heart_disease.csv and print a report.

Usage:
    python scripts/verify_data_profiler.py
    python scripts/verify_data_profiler.py --input fixtures/data/heart_disease.csv
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from satyarepro.tools.data.data_profiler import DataProfiler

_WIDTH = 60


def _header(title: str) -> None:
    print(f"\n{'═' * _WIDTH}")
    print(f"  {title}")
    print(f"{'═' * _WIDTH}")


def _section(title: str) -> None:
    print(f"\n── {title} {'─' * (_WIDTH - len(title) - 4)}")


def _print_report(data: dict) -> None:
    # 1. Basic stats
    _section("1. Basic Statistics")
    b = data["basic"]
    print(f"  Rows    : {b['rows']}")
    print(f"  Columns : {b['columns']}")
    print(f"  Dtypes  :")
    for col, dtype in b["dtypes"].items():
        print(f"    {col:<12} {dtype}")

    # 2. Missing values
    _section("2. Missing Values")
    if data["missing"]:
        for col, info in data["missing"].items():
            print(f"  {col:<12} {info['count']} missing  ({info['ratio']*100:.1f}%)")
        print(f"  {'Total':<12} {data['missing_total']}")
    else:
        print("  No missing values.")

    # 3. Class imbalance
    _section("3. Class Imbalance")
    ci = data["class_imbalance"]
    if "error" in ci:
        print(f"  {ci['error']}")
    else:
        print(f"  Column : {ci['column']}")
        for label, count in ci["distribution"].items():
            print(f"  Class {label:<5} {count} samples")
        ratio = ci["imbalance_ratio"]
        flag = "  ⚠ imbalanced" if ratio is not None and ratio >= 1.5 else ""
        print(f"  Imbalance ratio : {ratio}{flag}")

    # 4. Demographics
    _section("4. Demographic Distributions")
    demo = data["demographics"]
    if "sex" in demo:
        print("  Sex:")
        for val, count in demo["sex"]["distribution"].items():
            ratio = demo["sex"]["ratios"][val]
            label = demo["sex"]["labels"][val]
            print(f"    {label} ({val}) : {count}  ({ratio*100:.1f}%)")
    if "age" in demo:
        a = demo["age"]
        print("  Age (years):")
        print(f"    min={a['min']}  Q1={a['q25']}  median={a['median']}"
              f"  mean={a['mean']}  Q3={a['q75']}  max={a['max']}")
    if not demo:
        print("  No age/sex columns found.")

    # 5. Outliers (IQR)
    _section("5. Outliers (IQR method)")
    outlier_cols = {
        col: info for col, info in data["outliers"].items()
        if info["outlier_count"] > 0
    }
    if outlier_cols:
        for col, info in outlier_cols.items():
            print(f"  {col:<12} {info['outlier_count']} outliers  "
                  f"({info['outlier_ratio']*100:.1f}%)  "
                  f"fence [{info['lower_fence']}, {info['upper_fence']}]")
    else:
        print("  No outliers detected.")
    clean_cols = [c for c in data["outliers"] if data["outliers"][c]["outlier_count"] == 0]
    if clean_cols:
        print(f"  Clean columns : {', '.join(clean_cols)}")

    # 6. Data consistency
    _section("6. Data Consistency")
    cons = data["consistency"]
    if cons:
        for col, info in cons.items():
            n = info["out_of_range_count"]
            lo, hi = info["expected_range"]
            status = "OK" if n == 0 else f"⚠ {n} value(s) outside [{lo}, {hi}]"
            print(f"  {col:<12} expected [{lo}, {hi}]  → {status}")
    else:
        print("  No consistency checks configured for this dataset's columns.")


async def main(input_path: str) -> None:
    _header(f"DataProfiler Report")
    print(f"  Input : {input_path}")

    raw = await DataProfiler().execute(path=input_path)
    data = json.loads(raw)

    if "error" in data:
        print(f"\n  ERROR: {data['error']}")
        return

    _print_report(data)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile a CSV dataset with DataProfiler.")
    parser.add_argument(
        "--input",
        default="fixtures/data/heart_disease.csv",
        help="Path to CSV file (default: fixtures/data/heart_disease.csv).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.input))
