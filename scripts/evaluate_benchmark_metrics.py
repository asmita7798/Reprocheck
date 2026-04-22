#!/usr/bin/env python3
"""Compute per-item accuracy and pooled kappa for the 60-paper benchmark."""

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "manual_eval_comparison_60_smart_grobid.csv"
EVAL60_ROOT = ROOT / "sampled_papers_eval60"
OUTPUT_CSV = ROOT / "per_item_accuracy.csv"
POOLED_KAPPA_OUTPUT_CSV = ROOT / "pooled_kappa.csv"

ITEMS = [
    "dataset_info",
    "data_split",
    "hyperparameters",
    "hardware_info",
    "code_repo",
    "pseudocode",
    "theorems",
    "proofs",
    "method_well_described",
]

METHODS = [
    "pymupdf_8b",
    "pymupdf_70b",
    "smart_pymupdf_8b",
    "smart_pymupdf_70b",
    "smart_grobid_8b",
    "smart_grobid",
]

METHOD_FILE_PATTERNS = {
    "pymupdf_8b": "reproducibility_assessments_pymupdf_8b_eval60.csv",
    "pymupdf_70b": "reproducibility_assessments_pymupdf_70b_eval60.csv",
    "smart_pymupdf_8b": "reproducibility_assessments_smart_pymupdf_8b_eval60.csv",
    "smart_pymupdf_70b": "reproducibility_assessments_smart_pymupdf_70b_eval60.csv",
    "smart_grobid_8b": "reproducibility_assessments_grobid_smart_8b_eval60.csv",
}


def compute_cohens_kappa(human_labels: list[str], pred_labels: list[str]) -> float | None:
    valid_pairs = [
        (human_label, pred_label)
        for human_label, pred_label in zip(human_labels, pred_labels)
        if human_label in {"yes", "no"} and pred_label in {"yes", "no"}
    ]
    if not valid_pairs:
        return None

    n = len(valid_pairs)
    observed_agreement = sum(h == p for h, p in valid_pairs) / n
    human_yes = sum(h == "yes" for h, _ in valid_pairs) / n
    human_no = 1 - human_yes
    pred_yes = sum(p == "yes" for _, p in valid_pairs) / n
    pred_no = 1 - pred_yes
    expected_agreement = (human_yes * pred_yes) + (human_no * pred_no)

    if math.isclose(1 - expected_agreement, 0.0):
        return float("nan")
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def format_kappa(kappa: float | None) -> str:
    if kappa is None:
        return ""
    if math.isnan(kappa):
        return "nan"
    return f"{kappa:.4f}"


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paper_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["conference"], row["year"], row["file_name"])


def load_human_rows() -> list[dict[str, str]]:
    return load_csv_rows(INPUT_CSV)


def load_method_predictions(method: str, human_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    if method == "smart_grobid":
        predictions = {}
        for row in human_rows:
            predictions[paper_key(row)] = {
                item: row[f"smart_grobid_{item}_answer"].strip().lower()
                for item in ITEMS
            }
        return predictions

    filename = METHOD_FILE_PATTERNS[method]
    predictions: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in sorted(EVAL60_ROOT.glob(f"*/*/{filename}")):
        conference = path.parent.parent.name
        year = path.parent.name
        for row in load_csv_rows(path):
            predictions[(conference, year, row["file_name"])] = {
                item: row[f"{item}_answer"].strip().lower()
                for item in ITEMS
            }
    return predictions


def main() -> None:
    human_rows = load_human_rows()
    human_by_key = {paper_key(row): row for row in human_rows}
    method_predictions = {
        method: load_method_predictions(method, human_rows)
        for method in METHODS
    }

    output_rows = []
    for item in ITEMS:
        human_labels = [row[f"human_{item}"].strip().lower() for row in human_rows]
        result_row = {"item": item, "support": len(human_labels)}
        for method in METHODS:
            pred_labels = [
                method_predictions[method][paper_key(row)][item]
                for row in human_rows
            ]
            matches = sum(h == p for h, p in zip(human_labels, pred_labels))
            result_row[f"{method}_matches"] = matches
            result_row[f"{method}_accuracy"] = f"{matches / len(human_labels):.4f}"
        output_rows.append(result_row)

    fieldnames = ["item", "support"]
    for method in METHODS:
        fieldnames.extend([f"{method}_matches", f"{method}_accuracy"])

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    pooled_rows = []
    for method in METHODS:
        pooled_human_labels = []
        pooled_pred_labels = []
        excluded = 0
        for item in ITEMS:
            for row in human_rows:
                key = paper_key(row)
                human_label = row[f"human_{item}"].strip().lower()
                pred_label = method_predictions[method][key][item].strip().lower()
                if human_label in {"yes", "no"} and pred_label in {"yes", "no"}:
                    pooled_human_labels.append(human_label)
                    pooled_pred_labels.append(pred_label)
                else:
                    excluded += 1

        matches = sum(h == p for h, p in zip(pooled_human_labels, pooled_pred_labels))
        support = len(pooled_human_labels)
        pooled_rows.append(
            {
                "method": method,
                "support": support,
                "excluded": excluded,
                "matches": matches,
                "accuracy": f"{matches / support:.4f}" if support else "",
                "kappa": format_kappa(compute_cohens_kappa(pooled_human_labels, pooled_pred_labels)),
            }
        )

    with POOLED_KAPPA_OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "support", "excluded", "matches", "accuracy", "kappa"],
        )
        writer.writeheader()
        writer.writerows(pooled_rows)

    print(OUTPUT_CSV)
    print(POOLED_KAPPA_OUTPUT_CSV)


if __name__ == "__main__":
    main()
