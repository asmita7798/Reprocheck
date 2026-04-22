# Reprocheck

Reprocheck is an automated reproducibility-auditing pipeline that has been developed for AI conference papers.

Reproducibility is a central concern in modern AI and ML research, but auditing papers manually at scale is slow, labor-intensive, and difficult to standardize. Important signals such as dataset reporting, train/test splits, hyperparameters, hardware details, code availability, proofs, and method descriptions are often distributed unevenly across papers and expressed in inconsistent ways. Reprocheck addresses this by combining rule-based evidence retrieval with LLM-based item-level classification to audit reproducibility-related reporting directly from paper content.

The project compares multiple pipeline variants based on different text extraction and context construction strategies, including a baseline PyMuPDF pipeline, a Smart PyMuPDF pipeline, and a Smart GROBID pipeline. The final large-scale analysis reported in the project uses the Smart GROBID pipeline.

This repository includes the code needed for:

- paper-type classification
- PDF and TEI-based reproducibility auditing pipelines
- paper-level scoring
- benchmark evaluation
- report figure generation

The goal of the project is not to replace expert review, but to support large-scale meta-research on reporting practices across venues and over time.

## Benchmark Results

The pipeline variants were evaluated on a manually annotated benchmark of 60 papers. The strongest overall performance in the report was obtained by the Smart GROBID pipeline with the larger model.

| Pipeline | Accuracy | Cohen's Kappa |
| --- | ---: | ---: |
| PyMuPDF Pipeline (Llama 8b) | 0.5852 | 0.2456 |
| PyMuPDF Pipeline (Llama 70b) | 0.6963 | 0.4305 |
| Smart PyMuPDF Pipeline (Llama 8b) | 0.8444 | 0.6788 |
| Smart PyMuPDF Pipeline (Llama 70b) | 0.9056 | 0.8082 |
| Smart GROBID Pipeline (Llama 8b) | 0.8370 | 0.6652 |
| Smart GROBID Pipeline (Llama 70b) | 0.9204 | 0.8394 |

## Large-Scale Analysis

After model and pipeline selection, the Smart GROBID pipeline was applied to a larger conference-paper sample to study reproducibility reporting across venues and over time.

- Across the shared 2023-2025 comparison window, ACL and NeurIPS showed higher average reproducibility scores than AAAI.
- For NeurIPS, papers from the post-checklist period showed a higher score distribution than papers from the pre-checklist period.
- Code and data reporting patterns differed across venues, suggesting that reporting quality varies not only in overall score but also in the type of evidence papers provide.

Figure placeholders:

<p align="center">
  <img src="figures/final_venue_mean_repro_score_2023_2025.pdf" alt="Mean reproducibility score by conference" width="700">
</p>

<p align="center">
  <img src="figures/final_neurips_before_after_checklist_violin.pdf" alt="NeurIPS reproducibility scores before and after checklist adoption" width="700">
</p>

## Repository Structure

- `classify_paper_types.py`: classify papers as empirical, theoretical, or survey
- `extract_tei_with_grobid.py`: extract TEI XML from PDFs using GROBID
- `pipeline_pymupdf_baseline.py`: baseline pipeline using PDF text extraction
- `pipeline_smart_pymupdf.py`: structure-aware PyMuPDF pipeline
- `grobid_tei_utils.py`: shared TEI parsing helpers
- `pipeline_smart_grobid.py`: final Smart GROBID pipeline
- `score_smart_grobid.py`: compute weighted paper-level reproducibility scores
- `evaluate_benchmark_metrics.py`: compute benchmark accuracy and Cohen's kappa
- `notebooks/report_figures.ipynb`: generate the report figures

## Requirements

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

External tools:

- `GROBID` must be running locally for the TEI-based pipeline
- `PyMuPDF` is used for PDF extraction

## API Keys

The classification and pipeline scripts use OpenAI-compatible endpoints. Users must use their obtained API keys for running the scripts.

Example:

```bash
export GROQ_API_KEY=...
```

Some scripts also support:

```bash
export ACADEMIC_CLOUD_API_KEY=...
```

## Expected Data Layout

The scripts expect a directory structure like:

```text
sampled_papers/<conference>/<year>/sampled_papers.csv
sampled_papers/<conference>/<year>/papers/*.pdf
sampled_papers/<conference>/<year>/grobid_tei/*.tei.xml
sampled_papers_eval60/<conference>/<year>/...
```

## Workflow

1. Classify papers by type:

```bash
python classify_paper_types.py
```

2. Extract TEI XML:

```bash
python extract_tei_with_grobid.py
```

3. Run the final pipeline:

```bash
python pipeline_smart_grobid.py
```

4. Compute weighted paper scores:

```bash
python score_smart_grobid.py
```

5. Recompute benchmark metrics:

```bash
python evaluate_benchmark_metrics.py
```

6. Regenerate report figures:

```bash
jupyter lab notebooks/report_figures.ipynb
```

## Reproducing The Project

This repository does not include the full paper corpus or generated outputs. To reproduce the workflow, users should scrape or collect their own paper set, arrange it in the expected directory structure, and then run the scripts above.

The scraper used during this project was [ai_papers_scrapper](https://github.com/george-gca/ai_papers_scrapper).

## Notes

- Exact LLM outputs may vary across providers, model versions, and reruns.
- Scraped paper collections can differ across time, venues, PDF availability, and preprocessing quality, reproduced results may differ from those reported in the project.
