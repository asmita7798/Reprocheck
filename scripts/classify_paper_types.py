#!/usr/bin/env python3
"""Classify conference papers as empirical, theoretical, or survey."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_INPUT_ROOT = Path("sampled_papers")
DEFAULT_OUTPUT_NAME = "paper_classifications.csv"

SYSTEM_PROMPT = (
    "You are an expert peer reviewer for top-tier AI conferences. "
    "You must return valid JSON only."
)

USER_PROMPT_TEMPLATE = """You are an expert peer reviewer for top-tier AI conferences (NeurIPS, AAAI, ACL).
Analyze the provided paper title and abstract, then classify the paper into exactly one category based on its primary contribution.

Categories:
- Empirical: introduces a new model, algorithm, system, dataset, benchmark, or method and supports it mainly with experiments.
- Theoretical: focuses mainly on theorems, proofs, formal guarantees, complexity analysis, or mathematically derived results.
- Survey: mainly reviews prior work, presents a taxonomy, or argues a position/opinion without a primary new method or theorem.

Rules:
- Choose exactly one category.
- If a paper mixes theory and experiments, choose the category that reflects its main contribution.
- Return strict JSON only, with no markdown, no code fences, and no extra text.
- Confidence must be a float between 0 and 1.
- Reasoning must be one sentence with at most 25 words.

Paper Metadata:
Title: {title}
Abstract: {abstract}

Output format:
{{
  "classification": "Empirical",
  "confidence": 0.95,
  "reasoning": "The abstract proposes a new method and validates it experimentally on benchmark datasets."
}}"""

VALID_CLASSIFICATIONS = {"Empirical", "Theoretical", "Survey"}


class RateLimiter:
    """Sliding-window rate limiter for minute, hour, and day quotas."""

    def __init__(
        self,
        per_minute: int,
        per_hour: int,
        per_day: int,
    ) -> None:
        self.windows: list[tuple[float, int, deque[float]]] = [
            (60.0, per_minute, deque()),
            (3600.0, per_hour, deque()),
            (86400.0, per_day, deque()),
        ]

    def acquire(self) -> None:
        while True:
            now = time.time()
            wait_for = 0.0

            for window_seconds, limit, timestamps in self.windows:
                while timestamps and now - timestamps[0] >= window_seconds:
                    timestamps.popleft()

                if len(timestamps) >= limit:
                    wait_for = max(
                        wait_for,
                        window_seconds - (now - timestamps[0]),
                    )

            if wait_for <= 0:
                break

            sleep_for = max(wait_for, 0.01)
            print(
                "Rate limit reached; sleeping "
                f"{sleep_for:.1f}s to stay within API quotas."
            )
            time.sleep(sleep_for)

        request_time = time.time()
        for _window_seconds, _limit, timestamps in self.windows:
            timestamps.append(request_time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify sampled papers in each conference/year folder using an "
            "OpenAI-compatible LLM endpoint."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing sampled_papers/<conference>/<year>/sampled_papers.csv.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Name of the output CSV to create inside each conference/year folder.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GROQ_API_KEY"),
        help="API key for the OpenAI-compatible endpoint. Defaults to GROQ_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name to use for classification.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per paper if the API call or JSON parsing fails.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between successful requests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing classification CSVs instead of resuming from them.",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Optional cap on the number of papers to classify per folder for testing.",
    )
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=15,
        help="Maximum API requests allowed per minute.",
    )
    parser.add_argument(
        "--requests-per-hour",
        type=int,
        default=900,
        help="Maximum API requests allowed per hour.",
    )
    parser.add_argument(
        "--requests-per-day",
        type=int,
        default=21600,
        help="Maximum API requests allowed per day.",
    )
    return parser.parse_args()


def iter_sample_csvs(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("*/*/sampled_papers.csv"))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def strip_code_fences(text: str) -> str:
    fenced = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, flags=re.DOTALL)
    return fenced.group(1) if fenced else text.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_classification(value: str) -> str:
    normalized = value.strip().lower()
    mapping = {
        "empirical": "Empirical",
        "theoretical": "Theoretical",
        "survey": "Survey",
        "survey/position": "Survey",
        "survey or position": "Survey",
        "position": "Survey",
    }
    if normalized not in mapping:
        raise ValueError(f"Unexpected classification: {value!r}")
    return mapping[normalized]


def validate_response(payload: dict[str, Any]) -> dict[str, Any]:
    if "classification" not in payload or "confidence" not in payload or "reasoning" not in payload:
        raise ValueError("Response JSON is missing required keys.")

    classification = normalize_classification(str(payload["classification"]))
    confidence = float(payload["confidence"])
    reasoning = str(payload["reasoning"]).strip()

    if classification not in VALID_CLASSIFICATIONS:
        raise ValueError(f"Invalid classification: {classification}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence must be between 0 and 1. Got {confidence}")
    if not reasoning:
        raise ValueError("Reasoning cannot be empty.")

    return {
        "classification": classification,
        "confidence": confidence,
        "reasoning": reasoning,
    }


def classify_paper(
    client: OpenAI,
    model: str,
    temperature: float,
    title: str,
    abstract: str,
    max_retries: int,
    rate_limiter: RateLimiter | None = None,
) -> tuple[dict[str, Any], str]:
    prompt = USER_PROMPT_TEMPLATE.format(title=title, abstract=abstract)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            if rate_limiter is not None:
                rate_limiter.acquire()
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            payload = extract_json_object(content)
            return validate_response(payload), content
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(2**attempt)

    raise RuntimeError(f"Failed to classify paper after {max_retries} attempts: {last_error}")


def write_output_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "title",
        "file_name",
        "abstract",
        "classification",
        "confidence",
        "reasoning",
        "model",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit(
            "Missing API key. Pass --api-key or set GROQ_API_KEY."
        )

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    rate_limiter = RateLimiter(
        per_minute=args.requests_per_minute,
        per_hour=args.requests_per_hour,
        per_day=args.requests_per_day,
    )
    sample_csvs = iter_sample_csvs(args.input_root)
    if not sample_csvs:
        raise SystemExit(f"No sampled_papers.csv files found under {args.input_root}")

    for sample_csv in sample_csvs:
        output_csv = sample_csv.parent / args.output_name
        sampled_rows = load_csv_rows(sample_csv)
        if args.max_papers is not None:
            sampled_rows = sampled_rows[: args.max_papers]

        existing_rows = [] if args.overwrite else load_existing_rows(output_csv)
        completed_keys = {
            (row["title"], row["file_name"]) for row in existing_rows
        }
        output_rows = list(existing_rows)

        print(f"Processing {sample_csv.parent} ({len(sampled_rows)} papers)")

        for index, row in enumerate(sampled_rows, start=1):
            key = (row["title"], row["file_name"])
            if key in completed_keys:
                continue

            result, _raw_content = classify_paper(
                client=client,
                model=args.model,
                temperature=args.temperature,
                title=row["title"],
                abstract=row["abstract"],
                max_retries=args.max_retries,
                rate_limiter=rate_limiter,
            )

            output_rows.append(
                {
                    "title": row["title"],
                    "file_name": row["file_name"],
                    "abstract": row["abstract"],
                    "classification": result["classification"],
                    "confidence": result["confidence"],
                    "reasoning": result["reasoning"],
                    "model": args.model,
                }
            )
            write_output_rows(output_csv, output_rows)

            print(
                f"  [{index}/{len(sampled_rows)}] {row['file_name']} -> "
                f"{result['classification']} ({result['confidence']:.2f})"
            )

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
