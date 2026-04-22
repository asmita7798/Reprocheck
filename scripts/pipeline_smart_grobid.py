#!/usr/bin/env python3
"""Smart GROBID pipeline for reproducibility auditing of conference papers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from pipeline_pymupdf_baseline import (
    DEFAULT_CLASSIFICATION_NAME,
    DEFAULT_INPUT_NAME,
    DEFAULT_INPUT_ROOT,
    DEFAULT_LOG_DIRNAME,
    DEFAULT_MODEL,
    RUBRIC_ITEMS,
    HEURISTIC_PATTERNS,
    RateLimiter,
    build_text_blocks,
    clip_text,
    extract_json_object,
    filter_sample_csvs,
    iter_sample_csvs,
    load_classification_lookup,
    load_csv_rows,
    load_existing_rows,
    normalize_whitespace,
    resolve_abstract_text,
    setup_logger,
    split_paragraphs,
    write_raw_output,
)
from grobid_tei_utils import extract_filtered_tei_text, resolve_tei_path


DEFAULT_OUTPUT_NAME = "reproducibility_assessments_grobid_smart.csv"
DEFAULT_RAW_DIRNAME = "reproducibility_raw_grobid_smart"
DEFAULT_TEI_DIRNAME = "grobid_tei"
DEFAULT_FULL_TEXT_THRESHOLD = 50000
DEFAULT_MAX_CONTEXT_CHARS = 60000
DEFAULT_REGEX_SNIPPET_CHARS = 500
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
SECTION_PRIORITY = ("implementation", "experiments", "method", "data", "appendix", "introduction", "theory")

SYSTEM_PROMPT = (
    "You are an expert reproducibility auditor for AI research papers. "
    "Return strict JSON only. Do not guess. Base every answer on supplied evidence."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess reproducibility evidence from GROBID TEI XML using abstract-first plus smart full-text context."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--input-name", default=DEFAULT_INPUT_NAME)
    parser.add_argument("--classification-name", default=DEFAULT_CLASSIFICATION_NAME)
    parser.add_argument("--tei-dirname", default=DEFAULT_TEI_DIRNAME)
    parser.add_argument("--conference", default=None)
    parser.add_argument("--year", default=None)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--raw-dirname", default=DEFAULT_RAW_DIRNAME)
    parser.add_argument("--log-dirname", default=DEFAULT_LOG_DIRNAME)
    parser.add_argument("--api-key", default=os.environ.get("GROQ_API_KEY"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--full-text-threshold", type=int, default=DEFAULT_FULL_TEXT_THRESHOLD)
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    parser.add_argument("--regex-snippet-chars", type=int, default=DEFAULT_REGEX_SNIPPET_CHARS)
    parser.add_argument("--requests-per-minute", type=int, default=15)
    parser.add_argument("--requests-per-hour", type=int, default=900)
    parser.add_argument("--requests-per-day", type=int, default=21600)
    return parser.parse_args()


def build_prompt(title: str, abstract: str, context_text: str, pass_name: str) -> str:
    rubric_json = json.dumps(
        [{"key": item["key"], "question": item["question"]} for item in RUBRIC_ITEMS],
        indent=2,
    )
    return f"""Assess the reproducibility evidence of this research paper.

Paper metadata:
- Title: {title}
- Abstract: {abstract}

Assessment pass: {pass_name}

Rubric items:
{rubric_json}

Evidence from the paper:
{context_text}

Instructions:
- For every rubric item above, answer only "yes" or "no".
- Treat "yes" as: the paper itself contains enough evidence that the item is present.
- Treat "no" as: the item is absent, too weak, too indirect, or only implied.
- Use only the supplied evidence.
- Do not infer missing details.
- Keep each evidence field short, ideally under 40 words.
- Keep each reason field short, under 30 words.
- Return strict JSON only in this format:
{{
  "items": [
    {{
      "key": "dataset_info",
      "answer": "yes",
      "evidence": "Uses ImageNet and CIFAR-100 benchmarks.",
      "reason": "Datasets are named explicitly."
    }}
  ],
}}"""


def validate_items_only_response(payload: dict[str, Any]) -> dict[str, Any]:
    if "items" not in payload or not isinstance(payload["items"], list):
        raise ValueError("Response JSON is missing an items list.")

    expected_keys = {item["key"] for item in RUBRIC_ITEMS}
    normalized_items: list[dict[str, str]] = []
    seen_keys: set[str] = set()

    for item in payload["items"]:
        key = str(item.get("key", "")).strip()
        raw_answer = str(item.get("answer", "")).strip()
        answer = raw_answer.lower()
        evidence = str(item.get("evidence", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if key not in expected_keys:
            continue
        if answer in {"unclear", "partial", "partially", "somewhat", "mixed"}:
            answer = "no"
        elif answer not in {"yes", "no"}:
            answer = "no"
            fallback_note = f"Raw model answer: {raw_answer!r}."
            reason = f"{reason} {fallback_note}".strip() if reason else fallback_note
        normalized_items.append(
            {
                "key": key,
                "answer": answer,
                "evidence": evidence,
                "reason": reason,
            }
        )
        seen_keys.add(key)

    missing = expected_keys - seen_keys
    if missing:
        raise ValueError(f"Missing rubric items in response: {sorted(missing)}")

    return {
        "items": sorted(normalized_items, key=lambda item: item["key"]),
    }


def assess_with_llm(client: OpenAI, model: str, temperature: float, title: str, abstract: str, context_text: str, pass_name: str, max_retries: int, rate_limiter: RateLimiter | None) -> tuple[dict[str, Any], str]:
    prompt = build_prompt(title=title, abstract=abstract, context_text=context_text, pass_name=pass_name)
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
            payload = validate_items_only_response(extract_json_object(content))
            return payload, content
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to assess paper after {max_retries} attempts: {last_error}")


def collect_regex_snippets(text: str, snippet_chars: int) -> dict[str, list[str]]:
    normalized = normalize_whitespace(text)
    matches: dict[str, list[str]] = {}
    for item in RUBRIC_ITEMS:
        snippets: list[str] = []
        seen: set[str] = set()
        for pattern in HEURISTIC_PATTERNS.get(item["key"], []):
            for match in pattern.finditer(normalized):
                start = max(match.start() - snippet_chars // 2, 0)
                end = min(match.end() + snippet_chars // 2, len(normalized))
                snippet = clip_text(normalized[start:end], snippet_chars)
                if snippet and snippet not in seen:
                    seen.add(snippet)
                    snippets.append(snippet)
                if len(snippets) >= 3:
                    break
            if len(snippets) >= 3:
                break
        matches[item["key"]] = snippets
    return matches


def build_regex_context(regex_matches: dict[str, list[str]]) -> str:
    lines: list[str] = []
    for item in RUBRIC_ITEMS:
        snippets = regex_matches.get(item["key"], [])
        if not snippets:
            continue
        lines.append(f"{item['key']}:")
        for snippet in snippets:
            lines.append(f"- {snippet}")
    return "\n".join(lines)


def build_section_pruned_text(filtered_text: str, max_chars: int) -> tuple[str, int]:
    blocks = build_text_blocks(filtered_text)
    if not blocks:
        return filtered_text[:max_chars], 0

    ordered_blocks: list[str] = []
    seen_blocks: set[str] = set()
    for bucket in SECTION_PRIORITY:
        for block in blocks:
            if block["section_bucket"] != bucket:
                continue
            text = block["text"].strip()
            if not text or text in seen_blocks:
                continue
            seen_blocks.add(text)
            if block["heading"]:
                ordered_blocks.append(f"{block['heading']}\n{text}")
            else:
                ordered_blocks.append(text)

    used_chars = 0
    selected: list[str] = []
    for block in ordered_blocks:
        block_text = block.strip()
        if not block_text:
            continue
        extra = len(block_text) + 2
        if selected and used_chars + extra > max_chars:
            break
        if not selected and len(block_text) > max_chars:
            selected.append(block_text[:max_chars])
            used_chars = max_chars
            break
        selected.append(block_text)
        used_chars += extra
    return "\n\n".join(selected), len(selected)


def build_full_context(title: str, abstract: str, filtered_text: str, regex_matches: dict[str, list[str]], full_text_threshold: int, max_context_chars: int) -> dict[str, Any]:
    regex_context = build_regex_context(regex_matches)
    if len(filtered_text) <= full_text_threshold:
        text_context = filtered_text
        strategy = "full_filtered_text"
        selected_block_count = len(build_text_blocks(filtered_text))
    else:
        text_context, selected_block_count = build_section_pruned_text(filtered_text, max_context_chars)
        strategy = "section_pruned_full_text"

    pieces = [f"Title: {title}", f"Abstract: {abstract}"]
    if regex_context:
        pieces.append("Regex evidence hints\n" + regex_context)
    pieces.append("Filtered paper text\n" + text_context)
    context_text = "\n\n".join(piece for piece in pieces if piece.strip())
    return {
        "context_strategy": strategy,
        "context_text": context_text,
        "context_chars": len(context_text),
        "paragraph_count": len(split_paragraphs(filtered_text)),
        "block_count": len(build_text_blocks(filtered_text)),
        "selected_block_count": selected_block_count,
        "regex_matches": regex_matches,
        "regex_context": regex_context,
    }


def flatten_result(row: dict[str, str], pdf_path: Path, abstract: str, model: str, context_bundle: dict[str, Any], filtered_text: str, result: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {
        "title": row["title"],
        "file_name": row["file_name"],
        "abstract": abstract,
        "pdf_path": str(pdf_path),
        "model": model,
        "context_strategy": context_bundle["context_strategy"],
        "full_text_chars": len(filtered_text),
        "context_chars": context_bundle["context_chars"],
        "paragraph_count": context_bundle["paragraph_count"],
        "block_count": context_bundle["block_count"],
        "selected_block_count": context_bundle["selected_block_count"],
    }
    by_key = {item["key"]: item for item in result["items"]}
    for rubric in RUBRIC_ITEMS:
        item = by_key[rubric["key"]]
        flat[f"{rubric['key']}_answer"] = item["answer"]
        flat[f"{rubric['key']}_evidence"] = item["evidence"]
        flat[f"{rubric['key']}_reason"] = item["reason"]
    return flat


def output_fieldnames() -> list[str]:
    fields = [
        "title",
        "file_name",
        "abstract",
        "pdf_path",
        "model",
        "context_strategy",
        "full_text_chars",
        "context_chars",
        "paragraph_count",
        "block_count",
        "selected_block_count",
    ]
    for rubric in RUBRIC_ITEMS:
        key = rubric["key"]
        fields.extend([f"{key}_answer", f"{key}_evidence", f"{key}_reason"])
    return fields


def write_output_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    logger, log_path = setup_logger(args.input_root.parent / args.log_dirname)
    if not args.api_key:
        raise SystemExit("Missing API key. Pass --api-key or set ACADEMIC_CLOUD_API_KEY.")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    rate_limiter = RateLimiter(
        per_minute=args.requests_per_minute,
        per_hour=args.requests_per_hour,
        per_day=args.requests_per_day,
        logger=logger,
    )
    sample_csvs = filter_sample_csvs(
        iter_sample_csvs(args.input_root, args.input_name),
        conference=args.conference,
        year=args.year,
    )
    if not sample_csvs:
        raise SystemExit(
            f"No {args.input_name} files found under {args.input_root} "
            f"for conference={args.conference!r}, year={args.year!r}"
        )

    logger.info("Writing run log to %s", log_path)
    logger.info(
        "Starting smart GROBID run with conference=%s year=%s max_papers=%s overwrite=%s full_text_threshold=%s",
        args.conference,
        args.year,
        args.max_papers,
        args.overwrite,
        args.full_text_threshold,
    )

    for sample_csv in sample_csvs:
        output_csv = sample_csv.parent / args.output_name
        raw_dir = sample_csv.parent / args.raw_dirname
        classification_lookup = load_classification_lookup(sample_csv.parent / args.classification_name)
        sampled_rows = load_csv_rows(sample_csv)
        if args.max_papers is not None:
            sampled_rows = sampled_rows[: args.max_papers]

        existing_rows = [] if args.overwrite else load_existing_rows(output_csv)
        completed_keys = {(row["title"], row["file_name"]) for row in existing_rows}
        output_rows = list(existing_rows)

        logger.info("Processing %s (%d papers)", sample_csv.parent, len(sampled_rows))

        for index, row in enumerate(sampled_rows, start=1):
            key = (row["title"], row["file_name"])
            if key in completed_keys:
                continue

            tei_path = resolve_tei_path(sample_csv, args.tei_dirname, row)
            pdf_path = sample_csv.parent / "papers" / row["file_name"]
            abstract_text = ""
            filtered_payload: dict[str, Any] = {}
            abstract_result: dict[str, Any] | None = None
            abstract_output = ""
            full_result: dict[str, Any] | None = None
            full_output = ""
            context_bundle: dict[str, Any] = {
                "context_strategy": "",
                "context_text": "",
                "context_chars": 0,
                "paragraph_count": 0,
                "block_count": 0,
                "selected_block_count": 0,
                "regex_matches": {},
                "regex_context": "",
            }

            try:
                if not tei_path.exists():
                    raise FileNotFoundError(f"Missing TEI for row {row['file_name']}: {tei_path}")

                filtered_payload = extract_filtered_tei_text(tei_path)
                abstract_text = resolve_abstract_text(row, classification_lookup) or filtered_payload["abstract"]

                abstract_result, abstract_output = assess_with_llm(
                    client=client,
                    model=args.model,
                    temperature=args.temperature,
                    title=row["title"],
                    abstract=abstract_text,
                    context_text=f"Title: {row['title']}\n\nAbstract: {abstract_text}",
                    pass_name="abstract-first",
                    max_retries=args.max_retries,
                    rate_limiter=rate_limiter,
                )

                regex_matches = collect_regex_snippets(
                    filtered_payload["text"],
                    snippet_chars=args.regex_snippet_chars,
                )
                context_bundle = build_full_context(
                    title=row["title"],
                    abstract=abstract_text,
                    filtered_text=filtered_payload["text"],
                    regex_matches=regex_matches,
                    full_text_threshold=args.full_text_threshold,
                    max_context_chars=args.max_context_chars,
                )

                full_result, full_output = assess_with_llm(
                    client=client,
                    model=args.model,
                    temperature=args.temperature,
                    title=row["title"],
                    abstract=abstract_text,
                    context_text=context_bundle["context_text"],
                    pass_name=context_bundle["context_strategy"],
                    max_retries=args.max_retries,
                    rate_limiter=rate_limiter,
                )

                flat_row = flatten_result(
                    row=row,
                    pdf_path=pdf_path,
                    abstract=abstract_text,
                    model=args.model,
                    context_bundle=context_bundle,
                    filtered_text=filtered_payload["text"],
                    result=full_result,
                )
                output_rows.append(flat_row)
                write_output_rows(output_csv, output_rows)

                raw_payload = {
                    "paper": {
                        "title": row["title"],
                        "file_name": row["file_name"],
                        "tei_path": str(tei_path),
                        "abstract": abstract_text,
                    },
                    "filtered_text_chars": len(filtered_payload["text"]),
                    "kept_headings": filtered_payload["kept_headings"],
                    "dropped_headings": filtered_payload["dropped_headings"],
                    "kept_section_count": filtered_payload["kept_section_count"],
                    "dropped_section_count": filtered_payload["dropped_section_count"],
                    "footnote_count": filtered_payload["footnote_count"],
                    "abstract_model_output": abstract_output,
                    "abstract_result": abstract_result,
                    "context_strategy": context_bundle["context_strategy"],
                    "regex_matches": context_bundle["regex_matches"],
                    "regex_context": context_bundle["regex_context"],
                    "selected_context": context_bundle["context_text"],
                    "full_model_output": full_output,
                    "result": full_result,
                }
                write_raw_output(raw_dir / f"{Path(row['file_name']).stem}.json", raw_payload)

                yes_count = sum(1 for item in full_result["items"] if item["answer"] == "yes")
                logger.info(
                    "[%d/%d] %s -> %d/%d yes (%s)",
                    index,
                    len(sampled_rows),
                    row["file_name"],
                    yes_count,
                    len(RUBRIC_ITEMS),
                    context_bundle["context_strategy"],
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed processing [%d/%d] %s", index, len(sampled_rows), row["file_name"])
                error_payload = {
                    "paper": {
                        "title": row.get("title", ""),
                        "file_name": row.get("file_name", ""),
                        "tei_path": str(tei_path),
                        "abstract": abstract_text,
                    },
                    "error": str(exc),
                    "filtered_text_chars": len(filtered_payload.get("text", "")),
                    "kept_headings": filtered_payload.get("kept_headings", []),
                    "dropped_headings": filtered_payload.get("dropped_headings", []),
                    "abstract_model_output": abstract_output,
                    "context_strategy": context_bundle.get("context_strategy", ""),
                    "selected_context": context_bundle.get("context_text", ""),
                    "full_model_output": full_output,
                }
                write_raw_output(raw_dir / f"{Path(row['file_name']).stem}.error.json", error_payload)
                continue

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
