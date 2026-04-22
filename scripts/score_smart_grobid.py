#!/usr/bin/env python3
"""Compute weighted paper-level reproducibility scores from Smart GROBID outputs."""

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SAMPLED_PAPERS_ROOT = ROOT / "sampled_papers"
DEFAULT_OUTPUT_CSV = ROOT / "smart_grobid_repro_scores.csv"
DEFAULT_ASSESSMENT_FILENAME = "reproducibility_assessments_grobid_smart.csv"

SCORING_CONFIG = {
    "Empirical": {
        "included": True,
        "core_weights": {
            "dataset_info": 20,
            "data_split": 15,
            "hyperparameters": 10,
            "code_repo": 30,
            "method_well_described": 20,
            "hardware_info": 5,
        },
        "bonus_weights": {
            "pseudocode": 4,
            "theorems": 3,
            "proofs": 3,
        },
        "normalization_denominator": 110,
    },
    "Theoretical": {
        "included": True,
        "core_weights": {
            "proofs": 35,
            "method_well_described": 30,
            "pseudocode": 20,
            "code_repo": 10,
            "theorems": 5,
        },
        "bonus_weights": {
            "dataset_info": 4,
            "data_split": 2,
            "hyperparameters": 2,
            "hardware_info": 2,
        },
        "normalization_denominator": 110,
    },
    "Survey": {
        "included": False,
        "core_weights": {},
        "bonus_weights": {},
        "normalization_denominator": None,
    },
}

OUTPUT_FIELDS = [
    "conference",
    "year",
    "title",
    "file_name",
    "paper_type",
    "core_score",
    "bonus_score",
    "raw_total_score",
    "final_score",
    "scored",
]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def yes(answer: str) -> bool:
    return answer.strip().lower() == "yes"


def compute_weighted_score(
    assessment_row: dict[str, str],
    weights: dict[str, int],
) -> int:
    score = 0
    for item, weight in weights.items():
        if yes(assessment_row.get(f"{item}_answer", "")):
            score += weight
    return score


def build_output_row(
    conference: str,
    year: str,
    classification_row: dict[str, str],
    assessment_row: dict[str, str] | None,
) -> dict[str, str]:
    paper_type = classification_row["classification"].strip()
    config = SCORING_CONFIG.get(paper_type, SCORING_CONFIG["Survey"])

    row = {
        "conference": conference,
        "year": year,
        "title": classification_row["title"],
        "file_name": classification_row["file_name"],
        "paper_type": paper_type,
        "core_score": "",
        "bonus_score": "",
        "raw_total_score": "",
        "final_score": "",
        "scored": "no",
    }

    if not config["included"] or assessment_row is None:
        return row

    core_score = compute_weighted_score(assessment_row, config["core_weights"])
    bonus_score = compute_weighted_score(assessment_row, config["bonus_weights"])
    raw_total_score = core_score + bonus_score
    final_score = raw_total_score / config["normalization_denominator"]

    row.update(
        {
            "core_score": str(core_score),
            "bonus_score": str(bonus_score),
            "raw_total_score": str(raw_total_score),
            "final_score": f"{final_score:.4f}",
            "scored": "yes",
        }
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate smart_grobid reproducibility scores from per-conf/year assessment CSVs."
    )
    parser.add_argument(
        "--assessment-filename",
        default=DEFAULT_ASSESSMENT_FILENAME,
        help="Assessment CSV filename expected in each sampled_papers/<conf>/<year>/ directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_rows = []

    for classification_csv in sorted(SAMPLED_PAPERS_ROOT.glob("*/ */paper_classifications.csv".replace(" ", ""))):
        conference = classification_csv.parent.parent.name
        year = classification_csv.parent.name
        assessment_csv = classification_csv.parent / args.assessment_filename

        classification_rows = load_csv_rows(classification_csv)
        assessment_rows = load_csv_rows(assessment_csv)
        assessment_by_file = {row["file_name"]: row for row in assessment_rows}

        for classification_row in classification_rows:
            assessment_row = assessment_by_file.get(classification_row["file_name"])
            output_rows.append(
                build_output_row(
                    conference=conference,
                    year=year,
                    classification_row=classification_row,
                    assessment_row=assessment_row,
                )
            )

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(args.output)


if __name__ == "__main__":
    main()
