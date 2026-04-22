#!/usr/bin/env python3
"""Extract TEI XML from conference-paper PDFs using a local GROBID server."""

from __future__ import annotations

import argparse
import concurrent.futures
import mimetypes
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib import error, request


DEFAULT_INPUT_ROOT = Path("sampled_papers")
DEFAULT_SERVER_URL = "http://localhost:8070"
DEFAULT_API_PATH = "/api/processFulltextDocument"
DEFAULT_OUTPUT_DIRNAME = "grobid_tei"
DEFAULT_TIMEOUT = 300
DEFAULT_WORKERS = 2


@dataclass
class ExtractionResult:
    pdf_path: Path
    output_path: Path
    status: str
    detail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract TEI XML for PDFs under sampled_papers using GROBID."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing conference/year/papers folders.",
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help="Base URL for the running GROBID service.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of PDFs to process in parallel.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout per PDF in seconds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract PDFs even if the .tei.xml file already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N PDFs after sorting, useful for testing.",
    )
    return parser.parse_args()


def build_multipart_body(pdf_path: Path, field_name: str = "input") -> tuple[bytes, str]:
    boundary = f"----CodexBoundary{uuid.uuid4().hex}"
    mime_type = mimetypes.guess_type(pdf_path.name)[0] or "application/pdf"
    file_bytes = pdf_path.read_bytes()

    lines = [
        f"--{boundary}\r\n".encode("utf-8"),
        (
            f'Content-Disposition: form-data; name="{field_name}"; '
            f'filename="{pdf_path.name}"\r\n'
        ).encode("utf-8"),
        f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
        file_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(lines), boundary


def grobid_is_alive(server_url: str, timeout: int) -> bool:
    health_url = f"{server_url.rstrip('/')}/api/isalive"
    try:
        with request.urlopen(health_url, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def iter_pdf_paths(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.rglob("*.pdf") if "papers" in path.parts)


def output_path_for(pdf_path: Path) -> Path:
    papers_dir = pdf_path.parent
    year_dir = papers_dir.parent
    output_dir = year_dir / DEFAULT_OUTPUT_DIRNAME
    return output_dir / f"{pdf_path.stem}.tei.xml"


def extract_one(pdf_path: Path, server_url: str, timeout: int, overwrite: bool) -> ExtractionResult:
    output_path = output_path_for(pdf_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        return ExtractionResult(pdf_path, output_path, "skipped", "already exists")

    body, boundary = build_multipart_body(pdf_path)
    url = f"{server_url.rstrip('/')}{DEFAULT_API_PATH}"
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    try:
        with request.urlopen(req, timeout=timeout) as response:
            payload = response.read()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        return ExtractionResult(pdf_path, output_path, "failed", f"HTTP {exc.code}: {detail}")
    except Exception as exc:
        return ExtractionResult(pdf_path, output_path, "failed", str(exc))

    if not payload.strip():
        return ExtractionResult(pdf_path, output_path, "failed", "empty response body")

    output_path.write_bytes(payload)
    return ExtractionResult(pdf_path, output_path, "ok")


def summarize(results: Iterable[ExtractionResult]) -> tuple[int, int, int]:
    ok = skipped = failed = 0
    for result in results:
        if result.status == "ok":
            ok += 1
        elif result.status == "skipped":
            skipped += 1
        else:
            failed += 1
    return ok, skipped, failed


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()

    if not input_root.exists():
        print(f"Input root does not exist: {input_root}", file=sys.stderr)
        return 1

    if not grobid_is_alive(args.server_url, timeout=min(args.timeout, 10)):
        print(
            f"GROBID is not reachable at {args.server_url}. "
            "Start the Docker container first.",
            file=sys.stderr,
        )
        return 1

    pdf_paths = iter_pdf_paths(input_root)
    if args.limit is not None:
        pdf_paths = pdf_paths[: args.limit]

    if not pdf_paths:
        print(f"No PDFs found under {input_root}", file=sys.stderr)
        return 1

    total = len(pdf_paths)
    print(f"Found {total} PDFs under {input_root}")
    print(f"Writing TEI files into sibling '{DEFAULT_OUTPUT_DIRNAME}' folders")
    print(f"Using {args.workers} worker(s) against {args.server_url}")

    results: list[ExtractionResult] = []
    completed = 0
    lock = threading.Lock()
    start_time = time.time()

    def run_job(pdf_path: Path) -> ExtractionResult:
        return extract_one(
            pdf_path=pdf_path,
            server_url=args.server_url,
            timeout=args.timeout,
            overwrite=args.overwrite,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {executor.submit(run_job, pdf_path): pdf_path for pdf_path in pdf_paths}
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            with lock:
                results.append(result)
                completed += 1
                label = result.status.upper()
                detail = f" ({result.detail})" if result.detail else ""
                print(f"[{completed}/{total}] {label} {result.pdf_path.name}{detail}")

    ok, skipped, failed = summarize(results)
    elapsed = time.time() - start_time
    print(
        f"Finished in {elapsed:.1f}s | ok={ok} skipped={skipped} failed={failed}"
    )

    if failed:
        print("Failed files:")
        for result in results:
            if result.status == "failed":
                print(f"- {result.pdf_path} :: {result.detail}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
