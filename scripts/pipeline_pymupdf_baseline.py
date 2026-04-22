#!/usr/bin/env python3
"""PyMuPDF baseline pipeline for reproducibility auditing of conference papers."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_BASE_URL = "https://chat-ai.academiccloud.de/v1"
DEFAULT_MODEL = "meta-llama-3.1-8b-instruct"
DEFAULT_INPUT_ROOT = Path("sampled_papers")
DEFAULT_INPUT_NAME = "sampled_papers.csv"
DEFAULT_CLASSIFICATION_NAME = "paper_classifications.csv"
DEFAULT_OUTPUT_NAME = "reproducibility_assessments.csv"
DEFAULT_RAW_DIRNAME = "reproducibility_raw"
DEFAULT_LOG_DIRNAME = "logs"
DEFAULT_MAX_CONTEXT_CHARS = 70000 #24000 - prev
DEFAULT_SNIPPET_CHARS = 5000
DEFAULT_TOP_SNIPPETS_PER_ITEM = 7

SYSTEM_PROMPT = (
    "You are an expert reproducibility auditor for AI research papers. "
    "Return strict JSON only. Do not guess. Base every answer on supplied evidence."
)

RUBRIC_ITEMS: list[dict[str, Any]] = [
    {
        "key": "dataset_info",
        "question": "Does the paper clearly identify the dataset(s), data source, or benchmark used?",
        "keywords": [
            "dataset",
            "benchmark",
            "corpus",
            "data source",
            "evaluation data",
            "training data",
        ],
    },
    {
        "key": "data_split",
        "question": "Does the paper describe train/validation/test splits or another evaluation split protocol?",
        "keywords": [
            "train",
            "validation",
            "test",
            "dev set",
            "split",
            "cross-validation",
            "held-out",
        ],
    },
    {
        "key": "hyperparameters",
        "question": "Does the paper report concrete hyperparameters or optimization settings needed to reproduce results?",
        "keywords": [
            "hyperparameter",
            "learning rate",
            "batch size",
            "epoch",
            "dropout",
            "optimizer",
            "weight decay",
        ],
    },
    {
        "key": "hardware_info",
        "question": "Does the paper mention compute hardware or runtime environment details?",
        "keywords": [
            "gpu",
            "tpu",
            "cpu",
            "nvidia",
            "ram",
            "a100",
            "v100",
            "rtx",
            "memory",
            "hardware",
        ],
    },
    {
        "key": "code_repo",
        "question": "Does the paper provide a code repository, implementation URL, or explicit code availability statement?",
        "keywords": [
            "github",
            "gitlab",
            "code available",
            "implementation",
            "repository",
            "anonymous code",
            "supplementary material",
            "https://",
            "http://",
        ],
    },
    {
        "key": "pseudocode",
        "question": "Does the paper contain pseudocode, an algorithm block, or step-by-step procedural description?",
        "keywords": [
            "algorithm",
            "pseudo-code",
            "pseudocode",
            "procedure",
            "step 1",
            "step 2",
        ],
    },
    {
        "key": "theorems",
        "question": "Does the paper present theorem-like formal statements such as theorems, lemmas, propositions, or corollaries?",
        "keywords": [
            "theorem",
            "lemma",
            "proposition",
            "corollary",
        ],
    },
    {
        "key": "proofs",
        "question": "If theorem-like claims are present, does the paper provide proofs or proof sketches?",
        "keywords": [
            "proof",
            "proof sketch",
            "proved",
            "appendix",
        ],
    },
    {
        "key": "method_well_described",
        "question": "Is the proposed method described in enough detail that a technical reader could plausibly implement it?",
        "keywords": [
            "method",
            "approach",
            "architecture",
            "framework",
            "model",
            "training procedure",
            "experimental setup",
        ],
    },
]

RUBRIC_BY_KEY: dict[str, dict[str, Any]] = {item["key"]: item for item in RUBRIC_ITEMS}

APPLICABLE_RUBRIC_KEYS: dict[str, tuple[str, ...]] = {
    "Empirical": (
        "dataset_info",
        "data_split",
        "hyperparameters",
        "hardware_info",
        "code_repo",
        "method_well_described",
    ),
    "Theoretical": (
        "dataset_info",
        "pseudocode",
        "theorems",
        "proofs",
    ),
    "Survey": ("method_well_described",),
}

HEURISTIC_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "code_repo": [
        re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE),
        re.compile(r"\bgithub\.com/[^\s)>\]]+", re.IGNORECASE),
        re.compile(r"\bgitlab\.com/[^\s)>\]]+", re.IGNORECASE),
        re.compile(r"\banonymous[^.\n]{0,50}\bcode\b", re.IGNORECASE),
    ],
    "hardware_info": [
        re.compile(
            r"\b(?:gpu|gpus|tpu|tpus|cpu|cpus|nvidia|amd|ram|a100|v100|h100|rtx ?\d{3,4})\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b\d+\s*(?:gb|tb)\s*(?:ram|memory|vram)\b", re.IGNORECASE),
    ],
    "hyperparameters": [
        re.compile(
            r"\b(?:learning rate|batch size|epochs?|weight decay|dropout|optimizer|warmup)\b",
            re.IGNORECASE,
        ),
    ],
    "data_split": [
        re.compile(
            r"\b(?:train|training|validation|valid|dev|test|held-out|cross-validation)\b",
            re.IGNORECASE,
        ),
    ],
    "dataset_info": [
        re.compile(r"\b(?:dataset|datasets|benchmark|benchmarks|corpus|corpora)\b", re.IGNORECASE),
    ],
    "pseudocode": [
        re.compile(r"\b(?:algorithm\s*\d+|pseudocode|pseudo-code|procedure)\b", re.IGNORECASE),
    ],
    "theorems": [
        re.compile(r"\b(?:theorem|lemma|proposition|corollary)\b", re.IGNORECASE),
    ],
    "proofs": [
        re.compile(r"\b(?:proof|proof sketch|we prove)\b", re.IGNORECASE),
    ],
}

SECTION_BUCKET_KEYWORDS: dict[str, tuple[str, ...]] = {
    "abstract": ("abstract",),
    "introduction": ("introduction", "overview", "background"),
    "method": (
        "method",
        "methods",
        "proposed method",
        "proposed methods",
        "proposed approach",
        "our approach",
        "approach",
        "framework",
        "architecture",
        "model",
        "algorithm",
        "system",
    ),
    "data": (
        "dataset",
        "datasets",
        "data",
        "benchmark",
        "benchmarks",
        "corpus",
        "corpora",
        "preprocessing",
    ),
    "implementation": (
        "implementation",
        "implementation details",
        "training details",
        "training setup",
        "optimization",
        "hyperparameters",
        "compute",
        "hardware",
        "reproducibility",
    ),
    "experiments": (
        "experiments",
        "experimental setup",
        "experimental settings",
        "evaluation",
        "results",
        "ablation",
        "ablations",
        "analysis",
    ),
    "theory": (
        "theorem",
        "theorems",
        "proof",
        "proofs",
        "lemma",
        "lemmas",
        "proposition",
        "propositions",
        "corollary",
        "corollaries",
    ),
    "appendix": ("appendix", "appendices", "supplementary", "supplemental"),
    "references": ("references", "bibliography", "works cited"),
    "acknowledgements": ("acknowledg",),
}

RUBRIC_SECTION_PRIORITIES: dict[str, tuple[str, ...]] = {
    "dataset_info": ("data", "experiments", "method", "appendix"),
    "data_split": ("data", "experiments", "implementation", "appendix"),
    "hyperparameters": ("implementation", "experiments", "appendix", "method"),
    "hardware_info": ("implementation", "experiments", "appendix"),
    "code_repo": ("abstract", "introduction", "implementation", "appendix"),
    "pseudocode": ("method", "appendix"),
    "theorems": ("theory", "method", "appendix"),
    "proofs": ("theory", "appendix"),
    "method_well_described": ("method", "implementation", "experiments", "appendix"),
}

HEADING_NUMBER_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[IVXLCM]+|[A-Z])[\.\)]?\s+", re.IGNORECASE)
APPENDIX_PREFIX_RE = re.compile(r"^\s*appendix(?:\s+[A-Z0-9]+)?[\.\:]?\s*", re.IGNORECASE)


class RateLimiter:
    """Sliding-window rate limiter for minute, hour, and day quotas."""

    def __init__(
        self,
        per_minute: int,
        per_hour: int,
        per_day: int,
        logger: logging.Logger,
    ) -> None:
        self.windows: list[tuple[float, int, deque[float]]] = [
            (60.0, per_minute, deque()),
            (3600.0, per_hour, deque()),
            (86400.0, per_day, deque()),
        ]
        self.logger = logger

    def acquire(self) -> None:
        while True:
            now = time.time()
            wait_for = 0.0

            for window_seconds, limit, timestamps in self.windows:
                while timestamps and now - timestamps[0] >= window_seconds:
                    timestamps.popleft()
                if len(timestamps) >= limit:
                    wait_for = max(wait_for, window_seconds - (now - timestamps[0]))

            if wait_for <= 0:
                break

            sleep_for = max(wait_for, 0.01)
            self.logger.info(
                "Rate limit reached; sleeping %.1fs to stay within API quotas.",
                sleep_for,
            )
            time.sleep(sleep_for)

        request_time = time.time()
        for _window_seconds, _limit, timestamps in self.windows:
            timestamps.append(request_time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assess reproducibility evidence in sampled conference PDFs using "
            "regex heuristics plus an OpenAI-compatible LLM."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root containing sampled_papers/<conference>/<year>/sampled_papers.csv.",
    )
    parser.add_argument(
        "--input-name",
        default=DEFAULT_INPUT_NAME,
        help="Name of the sample metadata CSV within each conference/year folder.",
    )
    parser.add_argument(
        "--classification-name",
        default=DEFAULT_CLASSIFICATION_NAME,
        help="Optional paper classification CSV used to source abstract text when available.",
    )
    parser.add_argument(
        "--conference",
        default=None,
        help="Optional conference folder to process, for example aaai or neurips.",
    )
    parser.add_argument(
        "--year",
        default=None,
        help="Optional year folder to process, for example 2023.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Name of the reproducibility CSV written in each conference/year folder.",
    )
    parser.add_argument(
        "--raw-dirname",
        default=DEFAULT_RAW_DIRNAME,
        help="Directory name for per-paper raw JSON outputs.",
    )
    parser.add_argument(
        "--log-dirname",
        default=DEFAULT_LOG_DIRNAME,
        help="Directory name for run log files under the workspace root.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ACADEMIC_CLOUD_API_KEY"),
        help="API key for the OpenAI-compatible endpoint. Defaults to ACADEMIC_CLOUD_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name to use for reproducibility assessment.",
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
        help="Overwrite existing assessment CSVs instead of resuming from them.",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Optional cap on the number of papers to process per folder.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help="Character budget for the evidence context sent to the LLM.",
    )
    parser.add_argument(
        "--snippet-chars",
        type=int,
        default=DEFAULT_SNIPPET_CHARS,
        help="Maximum characters kept for a single evidence snippet.",
    )
    parser.add_argument(
        "--top-snippets-per-item",
        type=int,
        default=DEFAULT_TOP_SNIPPETS_PER_ITEM,
        help="Maximum retrieved snippets per rubric item before the LLM pass.",
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
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip the LLM call and output heuristic evidence only.",
    )
    return parser.parse_args()


def iter_sample_csvs(input_root: Path, input_name: str) -> list[Path]:
    return sorted(input_root.glob(f"*/*/{input_name}"))


def filter_sample_csvs(
    sample_csvs: list[Path],
    conference: str | None,
    year: str | None,
) -> list[Path]:
    filtered: list[Path] = []
    for path in sample_csvs:
        path_conference = path.parent.parent.name
        path_year = path.parent.name
        if conference and path_conference != conference:
            continue
        if year and path_year != year:
            continue
        filtered.append(path)
    return filtered


def setup_logger(log_dir: Path) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"reproducibility_run_{timestamp}.log"

    logger = logging.getLogger("reproducibility_check")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger, log_path


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_classification_lookup(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in load_csv_rows(path):
        lookup[(row.get("title", ""), row.get("file_name", ""))] = row
    return lookup


def strip_code_fences(text: str) -> str:
    fenced = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, flags=re.DOTALL)
    return fenced.group(1) if fenced else text.strip()


def extract_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def sanitize_json_like_text(text: str) -> str:
    sanitized = text.strip()
    sanitized = sanitized.replace("\u201c", '"').replace("\u201d", '"')
    sanitized = sanitized.replace("\u2018", "'").replace("\u2019", "'")
    sanitized = re.sub(r",\s*([}\]])", r"\1", sanitized)
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", sanitized)
    return sanitized


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        candidate = extract_balanced_json_object(cleaned)
        if candidate is None:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise
            candidate = match.group(0)

        candidate = sanitize_json_like_text(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not parse model JSON output after repair attempt: {candidate[:500]}"
            ) from exc


def normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        import fitz
    except ImportError:
        fitz = None

    if fitz is not None:
        pages: list[str] = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                pages.append(page.get_text("text"))
        return normalize_whitespace("\n\n".join(pages))

    pdftotext_path = shutil.which("pdftotext")
    if pdftotext_path:
        result = subprocess.run(
            [pdftotext_path, str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        return normalize_whitespace(result.stdout)

    raise RuntimeError(
        "PDF extraction requires PyMuPDF or pdftotext. Install PyMuPDF in the "
        "project environment, for example with: venv-reprocheck/bin/pip install pymupdf"
    )


def split_main_text_and_appendix_tail(text: str) -> tuple[str, str]:
    reference_match = re.search(
        r"(?im)^\s*(references|bibliography)\b.*$",
        text,
    )
    if not reference_match:
        return normalize_whitespace(text), ""

    main_text = normalize_whitespace(text[: reference_match.start()])
    tail = text[reference_match.end():].strip()
    if not tail:
        return main_text, ""

    checklist_match = re.search(r"(?im)^\s*checklist\b", tail)
    if checklist_match:
        tail = tail[: checklist_match.start()].strip()

    return main_text, normalize_whitespace(tail)


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return [p for p in paragraphs if len(p) > 20]


def clip_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def keyword_score(text: str, keywords: list[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(keyword.lower()) for keyword in keywords)


def normalize_heading_text(text: str) -> str:
    text = text.strip()
    text = HEADING_NUMBER_RE.sub("", text)
    text = APPENDIX_PREFIX_RE.sub("appendix ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .:-").lower()


def looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    if "\n" in stripped:
        return False
    lowered = stripped.lower()
    if lowered in {"references", "bibliography", "abstract", "appendix"}:
        return True
    if HEADING_NUMBER_RE.match(stripped) or APPENDIX_PREFIX_RE.match(stripped):
        return True
    words = stripped.split()
    if 1 <= len(words) <= 10 and stripped == stripped.title():
        return True
    if stripped.isupper() and len(words) <= 10:
        return True
    normalized = normalize_heading_text(stripped)
    return any(keyword in normalized for keywords in SECTION_BUCKET_KEYWORDS.values() for keyword in keywords)


def classify_heading_bucket(heading: str) -> str | None:
    normalized = normalize_heading_text(heading)
    for bucket, keywords in SECTION_BUCKET_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return bucket
    return None


def build_text_blocks(text: str) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    current_section = "body"
    current_heading = ""
    in_references = False

    for raw_block in re.split(r"\n\s*\n", text):
        block = raw_block.strip()
        if not block:
            continue

        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        first_line = lines[0]
        heading = ""
        content_lines = lines
        if looks_like_heading(first_line):
            heading = first_line
            content_lines = lines[1:]
            bucket = classify_heading_bucket(heading) or current_section
            current_heading = heading
            current_section = bucket
            in_references = bucket == "references"
            if not content_lines:
                continue

        if in_references and current_section == "references":
            if heading and classify_heading_bucket(heading) == "appendix":
                in_references = False
                current_section = "appendix"
            else:
                continue

        content = normalize_whitespace("\n".join(content_lines))
        if len(content) < 20:
            continue

        blocks.append(
            {
                "heading": current_heading,
                "section_bucket": current_section,
                "text": content,
            }
        )

    return blocks


def extract_appendix_tail_text(text: str) -> str:
    _main_text, appendix_tail = split_main_text_and_appendix_tail(text)
    return appendix_tail


def find_heading_blocks(blocks: list[dict[str, str]]) -> list[dict[str, str]]:
    preferred_buckets = {"abstract", "introduction", "method", "implementation", "experiments", "theory", "appendix"}
    return [block for block in blocks if block["section_bucket"] in preferred_buckets][:12]


def collect_regex_matches(text: str, item_key: str, snippet_chars: int, max_matches: int) -> list[str]:
    snippets: list[str] = []
    seen: set[str] = set()

    for pattern in HEURISTIC_PATTERNS.get(item_key, []):
        for match in pattern.finditer(text):
            start = max(match.start() - snippet_chars // 2, 0)
            end = min(match.end() + snippet_chars // 2, len(text))
            snippet = normalize_whitespace(text[start:end])
            snippet = clip_text(snippet, snippet_chars)
            if snippet and snippet not in seen:
                snippets.append(snippet)
                seen.add(snippet)
            if len(snippets) >= max_matches:
                return snippets
    return snippets


def score_block_for_item(block: dict[str, str], item: dict[str, Any], prefer_appendix: bool) -> int:
    score = keyword_score(block["text"], item["keywords"]) * 10
    section_bucket = block["section_bucket"]
    priorities = RUBRIC_SECTION_PRIORITIES.get(item["key"], ())
    if section_bucket in priorities:
        score += 15 - priorities.index(section_bucket) * 3
    if prefer_appendix and section_bucket == "appendix":
        score += 8
    if block["heading"]:
        normalized_heading = normalize_heading_text(block["heading"])
        if any(keyword in normalized_heading for keyword in item["keywords"]):
            score += 8
    return score


def build_context_bundle(
    title: str,
    abstract: str,
    full_text: str,
    max_context_chars: int,
    snippet_chars: int,
    top_snippets_per_item: int,
    target_keys: set[str] | None = None,
    prefer_appendix: bool = False,
) -> dict[str, Any]:
    source_blocks = build_text_blocks(full_text)
    searchable_text = "\n\n".join(block["text"] for block in source_blocks)
    selected_blocks: list[str] = []
    seen: set[str] = set()

    def add_block(block: str) -> None:
        normalized = clip_text(normalize_whitespace(block), snippet_chars)
        if normalized and normalized not in seen:
            selected_blocks.append(normalized)
            seen.add(normalized)

    add_block(f"Title: {title}")
    add_block(f"Abstract: {abstract}")

    for block in find_heading_blocks(source_blocks):
        prefix = f"[{block['section_bucket']}] "
        add_block(prefix + block["text"])

    active_items = [item for item in RUBRIC_ITEMS if target_keys is None or item["key"] in target_keys]
    heuristic_matches: dict[str, list[str]] = {}
    per_item_context: dict[str, list[str]] = {}
    for item in active_items:
        key = item["key"]
        scored_blocks = sorted(
            source_blocks,
            key=lambda block: score_block_for_item(block, item, prefer_appendix=prefer_appendix),
            reverse=True,
        )
        top_scored = [
            block for block in scored_blocks if score_block_for_item(block, item, prefer_appendix=prefer_appendix) > 0
        ][:top_snippets_per_item]
        item_context: list[str] = []
        for block in top_scored:
            snippet = f"[{block['section_bucket']}] {block['text']}"
            item_context.append(clip_text(snippet, snippet_chars))
            add_block(snippet)
        regex_snippets = collect_regex_matches(
            searchable_text,
            key,
            snippet_chars,
            max_matches=top_snippets_per_item,
        )
        heuristic_matches[key] = regex_snippets
        per_item_context[key] = item_context + regex_snippets
        for snippet in regex_snippets:
            add_block(snippet)

    context_lines: list[str] = []
    used_chars = 0
    for block in selected_blocks:
        candidate = f"- {block}"
        if used_chars + len(candidate) + 1 > max_context_chars:
            break
        context_lines.append(candidate)
        used_chars += len(candidate) + 1

    return {
        "context_text": "\n".join(context_lines),
        "heuristics": heuristic_matches,
        "per_item_context": per_item_context,
        "paragraph_count": len(split_paragraphs(full_text)),
        "block_count": len(source_blocks),
        "selected_block_count": len(context_lines),
    }


def build_abstract_context_bundle(
    title: str,
    abstract: str,
    snippet_chars: int,
    target_keys: set[str] | None = None,
) -> dict[str, Any]:
    selected_blocks: list[str] = []
    seen: set[str] = set()

    def add_block(block: str) -> None:
        normalized = clip_text(normalize_whitespace(block), snippet_chars)
        if normalized and normalized not in seen:
            selected_blocks.append(normalized)
            seen.add(normalized)

    add_block(f"Title: {title}")
    add_block(f"Abstract: {abstract}")

    active_items = [item for item in RUBRIC_ITEMS if target_keys is None or item["key"] in target_keys]
    heuristic_matches: dict[str, list[str]] = {}
    per_item_context: dict[str, list[str]] = {}
    normalized_abstract = normalize_whitespace(abstract)

    for item in active_items:
        key = item["key"]
        heuristic_matches[key] = collect_regex_matches(
            normalized_abstract,
            key,
            snippet_chars,
            max_matches=2,
        )
        item_context: list[str] = []
        if keyword_score(normalized_abstract, item["keywords"]) > 0:
            item_context.append(clip_text(f"[abstract] {normalized_abstract}", snippet_chars))
            add_block(f"[abstract] {normalized_abstract}")
        for snippet in heuristic_matches[key]:
            add_block(snippet)
        per_item_context[key] = item_context + heuristic_matches[key]

    return {
        "context_text": "\n".join(f"- {block}" for block in selected_blocks),
        "heuristics": heuristic_matches,
        "per_item_context": per_item_context,
        "paragraph_count": 1 if normalized_abstract else 0,
        "block_count": 1 if normalized_abstract else 0,
        "selected_block_count": len(selected_blocks),
    }


def build_user_prompt(
    title: str,
    abstract: str,
    context_text: str,
    rubric_items: list[dict[str, Any]] | None = None,
    pass_name: str = "initial",
) -> str:
    active_items = rubric_items or RUBRIC_ITEMS
    rubric_json = json.dumps(
        [{"key": item["key"], "question": item["question"]} for item in active_items],
        indent=2,
    )
    return f"""Assess the reproducibility evidence of this research paper.

Paper metadata:
- Title: {title}
- Abstract: {abstract}

Assessment pass: {pass_name}

Rubric items:
{rubric_json}

Evidence snippets from the paper:
{context_text}

Instructions:
- For each rubric item, answer only "yes" or "no".
- Return only the rubric items listed above. Do not add extra rubric keys.
- Use only the supplied evidence snippets.
- If the evidence is weak, indirect, ambiguous, or missing, answer "no".
- Do not infer that something exists unless it is actually evidenced.
- Keep each evidence field short, ideally a quote fragment or concise paraphrase under 35 words.
- Keep each reason short, under 30 words.
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
  "summary": "Short summary of the paper's reproducibility signals."
}}"""


def validate_response(payload: dict[str, Any], rubric_items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if "items" not in payload or "summary" not in payload:
        raise ValueError("Response JSON is missing required keys.")

    items = payload["items"]
    if not isinstance(items, list):
        raise ValueError("items must be a list.")

    expected_keys = {item["key"] for item in (rubric_items or RUBRIC_ITEMS)}
    normalized_items: list[dict[str, str]] = []
    seen_keys: set[str] = set()

    for item in items:
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
            raise ValueError(f"Invalid answer {answer!r} for rubric item {key!r}")
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

    summary = str(payload["summary"]).strip()
    return {
        "items": sorted(normalized_items, key=lambda item: item["key"]),
        "summary": summary,
    }


def assess_with_llm(
    client: OpenAI,
    model: str,
    temperature: float,
    title: str,
    abstract: str,
    context_text: str,
    max_retries: int,
    rate_limiter: RateLimiter | None = None,
    rubric_items: list[dict[str, Any]] | None = None,
    pass_name: str = "initial",
) -> tuple[dict[str, Any], str]:
    prompt = build_user_prompt(
        title=title,
        abstract=abstract,
        context_text=context_text,
        rubric_items=rubric_items,
        pass_name=pass_name,
    )
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
            return validate_response(payload, rubric_items=rubric_items), content
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(2**attempt)

    raise RuntimeError(f"Failed to assess paper after {max_retries} attempts: {last_error}")


def heuristic_only_result(
    heuristics: dict[str, list[str]],
) -> dict[str, Any]:
    items: list[dict[str, str]] = []
    for rubric in RUBRIC_ITEMS:
        key = rubric["key"]
        evidence_list = heuristics.get(key, [])
        answer = "yes" if evidence_list else "no"
        items.append(
            {
                "key": key,
                "answer": answer,
                "evidence": evidence_list[0] if evidence_list else "",
                "reason": (
                    "Matched heuristic pattern in extracted text."
                    if evidence_list
                    else "No heuristic match; LLM assessment skipped."
                ),
            }
        )

    return {
        "items": items,
        "summary": "Heuristic-only run; treat results as weak signals until LLM verification.",
    }


def merge_results(
    base_result: dict[str, Any],
    focused_result: dict[str, Any],
) -> dict[str, Any]:
    merged_by_key = {item["key"]: item for item in base_result["items"]}
    for item in focused_result["items"]:
        merged_by_key[item["key"]] = item

    merged_items = [merged_by_key[item["key"]] for item in RUBRIC_ITEMS]
    return {
        "items": merged_items,
        "summary": focused_result["summary"] or base_result["summary"],
    }


def find_unclear_keys(result: dict[str, Any]) -> set[str]:
    return set()


def find_second_pass_keys(result: dict[str, Any]) -> set[str]:
    return {
        item["key"]
        for item in result["items"]
        if item["answer"] == "no"
    }


def resolve_abstract_text(
    row: dict[str, str],
    classification_lookup: dict[tuple[str, str], dict[str, str]],
) -> str:
    key = (row.get("title", ""), row.get("file_name", ""))
    classification_row = classification_lookup.get(key, {})
    abstract = classification_row.get("abstract") or row.get("abstract") or ""
    return str(abstract).strip().strip("\"'")


def resolve_paper_classification(
    row: dict[str, str],
    classification_lookup: dict[tuple[str, str], dict[str, str]],
) -> str:
    key = (row.get("title", ""), row.get("file_name", ""))
    classification_row = classification_lookup.get(key, {})
    classification = str(classification_row.get("classification", "")).strip().lower()
    mapping = {
        "empirical": "Empirical",
        "theoretical": "Theoretical",
        "survey": "Survey",
    }
    return mapping.get(classification, "Empirical")


def flatten_result(
    row: dict[str, str],
    pdf_path: Path,
    text_stats: dict[str, Any],
    result: dict[str, Any],
    model: str,
    classification: str,
) -> dict[str, Any]:
    flat: dict[str, Any] = {
        "title": row["title"],
        "file_name": row["file_name"],
        "abstract": row["abstract"],
        "pdf_path": str(pdf_path),
        "model": model,
        "paper_classification": classification,
        "full_text_chars": text_stats["full_text_chars"],
        "paragraph_count": text_stats["paragraph_count"],
        "block_count": text_stats["block_count"],
        "selected_block_count": text_stats["selected_block_count"],
        "summary": result["summary"],
    }
    by_key = {item["key"]: item for item in result["items"]}
    for rubric in RUBRIC_ITEMS:
        key = rubric["key"]
        item = by_key[key]
        flat[f"{key}_answer"] = item["answer"]
        flat[f"{key}_evidence"] = item["evidence"]
        flat[f"{key}_reason"] = item["reason"]
    return flat


def output_fieldnames() -> list[str]:
    fields = [
        "title",
        "file_name",
        "abstract",
        "pdf_path",
        "model",
        "paper_classification",
        "full_text_chars",
        "paragraph_count",
        "block_count",
        "selected_block_count",
        "summary",
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


def write_raw_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def resolve_pdf_path(sample_csv: Path, row: dict[str, str]) -> Path:
    return sample_csv.parent / "papers" / row["file_name"]


def main() -> None:
    args = parse_args()
    logger, log_path = setup_logger(args.input_root.parent / args.log_dirname)
    if not args.skip_llm and not args.api_key:
        raise SystemExit("Missing API key. Pass --api-key or set ACADEMIC_CLOUD_API_KEY.")

    client = None if args.skip_llm else OpenAI(api_key=args.api_key, base_url=args.base_url)
    rate_limiter = RateLimiter(
        per_minute=args.requests_per_minute,
        per_hour=args.requests_per_hour,
        per_day=args.requests_per_day,
        logger=logger,
    )
    sample_csvs = iter_sample_csvs(args.input_root, args.input_name)
    sample_csvs = filter_sample_csvs(
        sample_csvs,
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
        "Starting run with conference=%s year=%s max_papers=%s skip_llm=%s overwrite=%s",
        args.conference,
        args.year,
        args.max_papers,
        args.skip_llm,
        args.overwrite,
    )

    for sample_csv in sample_csvs:
        output_csv = sample_csv.parent / args.output_name
        raw_dir = sample_csv.parent / args.raw_dirname
        classification_lookup = load_classification_lookup(
            sample_csv.parent / args.classification_name
        )
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
            pdf_path = resolve_pdf_path(sample_csv, row)
            abstract_text = ""
            paper_classification = ""
            abstract_response = ""
            first_pass_context = ""
            first_pass_response = ""
            second_pass_context = ""
            second_pass_response = ""
            raw_response = ""
            context_bundle = {
                "context_text": "",
                "heuristics": {},
                "per_item_context": {},
                "paragraph_count": 0,
                "block_count": 0,
                "selected_block_count": 0,
            }
            abstract_bundle = {
                "context_text": "",
                "heuristics": {},
            }
            full_text = ""
            main_text = ""
            appendix_tail_text = ""

            try:
                if not pdf_path.exists():
                    raise FileNotFoundError(f"Missing PDF for row {row['file_name']}: {pdf_path}")

                abstract_text = resolve_abstract_text(row, classification_lookup)
                paper_classification = resolve_paper_classification(row, classification_lookup)
                active_rubric_items = RUBRIC_ITEMS
                active_rubric_keys = {item["key"] for item in active_rubric_items}
                abstract_bundle = build_abstract_context_bundle(
                    title=row["title"],
                    abstract=abstract_text,
                    snippet_chars=args.snippet_chars,
                    target_keys=active_rubric_keys,
                )
                first_pass_bundle = {"heuristics": {}}
                second_pass_bundle = {"heuristics": {}}

                if args.skip_llm:
                    heuristic_map = dict(abstract_bundle["heuristics"])
                    unresolved_keys = {key for key in active_rubric_keys if not heuristic_map.get(key)}

                    if unresolved_keys:
                        full_text = extract_pdf_text(pdf_path)
                        main_text, appendix_tail_text = split_main_text_and_appendix_tail(full_text)
                        context_bundle = build_context_bundle(
                            title=row["title"],
                            abstract=abstract_text,
                            full_text=main_text,
                            max_context_chars=args.max_context_chars,
                            snippet_chars=args.snippet_chars,
                            top_snippets_per_item=args.top_snippets_per_item,
                            target_keys=unresolved_keys,
                        )
                        first_pass_bundle = context_bundle
                        for key_name in unresolved_keys:
                            heuristic_map[key_name] = context_bundle["heuristics"].get(key_name, [])

                    result = heuristic_only_result(
                        heuristic_map,
                    )
                    model_name = "heuristic-only"
                else:
                    result, abstract_response = assess_with_llm(
                        client=client,
                        model=args.model,
                        temperature=args.temperature,
                        title=row["title"],
                        abstract=abstract_text,
                        context_text=abstract_bundle["context_text"],
                        max_retries=args.max_retries,
                        rate_limiter=rate_limiter,
                        rubric_items=active_rubric_items,
                        pass_name="abstract-first-pass",
                    )
                    model_name = args.model
                    unclear_keys = find_unclear_keys(result)
                    if unclear_keys:
                        full_text = extract_pdf_text(pdf_path)
                        main_text, appendix_tail_text = split_main_text_and_appendix_tail(full_text)
                        focused_items = [item for item in RUBRIC_ITEMS if item["key"] in unclear_keys]
                        first_pass_bundle = build_context_bundle(
                            title=row["title"],
                            abstract=abstract_text,
                            full_text=main_text,
                            max_context_chars=args.max_context_chars,
                            snippet_chars=args.snippet_chars,
                            top_snippets_per_item=args.top_snippets_per_item,
                            target_keys=unclear_keys,
                            prefer_appendix=False,
                        )
                        first_pass_context = first_pass_bundle["context_text"]
                        if first_pass_context.strip():
                            focused_result, first_pass_response = assess_with_llm(
                                client=client,
                                model=args.model,
                                temperature=args.temperature,
                                title=row["title"],
                                abstract=abstract_text,
                                context_text=first_pass_context,
                                max_retries=args.max_retries,
                                rate_limiter=rate_limiter,
                                rubric_items=focused_items,
                                pass_name="full-text-first-pass",
                            )
                            result = merge_results(result, focused_result)
                            context_bundle = first_pass_bundle

                    second_pass_keys = find_second_pass_keys(result)
                    if second_pass_keys:
                        if not full_text:
                            full_text = extract_pdf_text(pdf_path)
                            main_text, appendix_tail_text = split_main_text_and_appendix_tail(full_text)
                        focused_items = [item for item in RUBRIC_ITEMS if item["key"] in second_pass_keys]
                        appendix_tail_text = extract_appendix_tail_text(full_text)
                        second_pass_bundle = build_context_bundle(
                            title=row["title"],
                            abstract=abstract_text,
                            full_text=appendix_tail_text or main_text or full_text,
                            max_context_chars=args.max_context_chars,
                            snippet_chars=args.snippet_chars,
                            top_snippets_per_item=max(args.top_snippets_per_item + 1, args.top_snippets_per_item),
                            target_keys=second_pass_keys,
                            prefer_appendix=True,
                        )
                        second_pass_context = second_pass_bundle["context_text"]
                        if second_pass_context.strip():
                            focused_result, second_pass_response = assess_with_llm(
                                client=client,
                                model=args.model,
                                temperature=args.temperature,
                                title=row["title"],
                                abstract=abstract_text,
                                context_text=second_pass_context,
                                max_retries=args.max_retries,
                                rate_limiter=rate_limiter,
                                rubric_items=focused_items,
                                pass_name="focused-second-pass",
                            )
                            result = merge_results(result, focused_result)
                            context_bundle = second_pass_bundle
                    raw_response = first_pass_response

                text_stats = {
                    "full_text_chars": len(full_text),
                    "paragraph_count": context_bundle["paragraph_count"],
                    "block_count": context_bundle["block_count"],
                    "selected_block_count": context_bundle["selected_block_count"],
                }
                flat_row = flatten_result(
                    row=row,
                    pdf_path=pdf_path,
                    text_stats=text_stats,
                    result=result,
                    model=model_name,
                    classification=paper_classification,
                )
                flat_row["abstract"] = abstract_text
                output_rows.append(flat_row)
                write_output_rows(output_csv, output_rows)

                raw_payload = {
                    "paper": {
                        "title": row["title"],
                        "file_name": row["file_name"],
                        "pdf_path": str(pdf_path),
                        "abstract": abstract_text,
                    },
                    "main_text_chars": len(main_text),
                    "appendix_tail_chars": len(appendix_tail_text),
                    "abstract_context": abstract_bundle["context_text"],
                    "abstract_heuristics": abstract_bundle["heuristics"],
                    "abstract_model_output": abstract_response if not args.skip_llm else "",
                    "heuristics": context_bundle["heuristics"],
                    "selected_context": context_bundle["context_text"],
                    "per_item_context": context_bundle["per_item_context"],
                    "first_pass_context": first_pass_context,
                    "first_pass_model_output": first_pass_response,
                    "second_pass_context": second_pass_context,
                    "result": result,
                    "raw_model_output": raw_response,
                    "second_pass_model_output": second_pass_response,
                }
                write_raw_output(raw_dir / f"{pdf_path.stem}.json", raw_payload)

                logger.info(
                    "[%d/%d] %s processed",
                    index,
                    len(sampled_rows),
                    row["file_name"],
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Failed processing [%d/%d] %s",
                    index,
                    len(sampled_rows),
                    row["file_name"],
                )
                error_payload = {
                    "paper": {
                        "title": row.get("title", ""),
                        "file_name": row.get("file_name", ""),
                        "pdf_path": str(pdf_path),
                        "abstract": abstract_text,
                    },
                    "error": str(exc),
                    "abstract_context": abstract_bundle.get("context_text", ""),
                    "abstract_model_output": abstract_response,
                    "first_pass_context": first_pass_context,
                    "first_pass_model_output": first_pass_response,
                    "second_pass_context": second_pass_context,
                    "second_pass_model_output": second_pass_response,
                }
                write_raw_output(raw_dir / f"{pdf_path.stem}.error.json", error_payload)
                continue

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
