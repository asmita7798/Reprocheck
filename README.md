# Automated Verification of Reproducibility Reporting in AI Research

This repository contains the code and analysis pipeline for automated verification of reproducibility-related reporting in AI research papers.

The framework combines structured document extraction, evidence retrieval, and LLM-based classification to assess whether papers report key reproducibility-related information, including dataset information, data splits, hyperparameters, hardware details, code repositories, pseudocode, theorems, proofs, and method descriptions.

The project evaluates multiple pipeline variants based on different extraction and context-construction strategies, including a baseline PyMuPDF pipeline, a Smart PyMuPDF pipeline, and a Smart GROBID pipeline.

The goal of the project is not to replace expert review, but to support large-scale assessment of reproducibility reporting in scientific papers.

---

## Features

- Automated verification of reproducibility reporting in AI papers using structured document extraction and LLM-based item-level classification
- Benchmark evaluation against manually annotated papers
- Large-scale analysis across AAAI, ACL, and NeurIPS papers

---

### API Configuration

The pipeline uses OpenAI-compatible API endpoints for LLM-based classification and auditing.

Set the required API keys as environment variables depending on the provider being used.

Example:

```bash
export GROQ_API_KEY=...
export ACADEMIC_CLOUD_API_KEY=...
```

The scripts can be adapted to other compatible providers and models by modifying the corresponding API configuration in the pipeline scripts.

---

## Project Structure

```text
reprocheck/
├── figures/
│   ├── final_neurips_before_after_checklist_*.png
│   └── final_venue_mean_repro_score_*.png
│
├── scripts/
│   ├── classify_paper_types.py
│   ├── evaluate_benchmark_metrics.py
│   ├── extract_tei_with_grobid.py
│   ├── pipeline_pymupdf_baseline.py
│   ├── pipeline_smart_grobid.py
│   ├── pipeline_smart_pymupdf.py
│   ├── report_figures.ipynb
│   └── score_smart_grobid.py
│
├── README.md
└── requirements.txt
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd reprocheck
```

### 2. Set Up Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure API Keys

Set the required API keys as environment variables:

```bash
export GROQ_API_KEY=...
export ACADEMIC_CLOUD_API_KEY=...
```

### 4. Prepare Data

The scripts expect a directory structure similar to:

```text
sampled_papers/<conference>/<year>/sampled_papers.csv
sampled_papers/<conference>/<year>/papers/*.pdf
sampled_papers/<conference>/<year>/grobid_tei/*.tei.xml
sampled_papers_eval60/<conference>/<year>/...
```

### 5. Run the Reproducibility Reporting Pipeline

#### Paper-Type Classification

```bash
python scripts/classify_paper_types.py
```

#### Structured Document Extraction

```bash
python scripts/extract_tei_with_grobid.py
```

#### Reproducibility Reporting Pipeline

```bash
python scripts/pipeline_smart_grobid.py
```

#### Reproducibility Reporting Scoring

```bash
python scripts/score_smart_grobid.py
```

### 6. Evaluate Benchmark Performance

```bash
python scripts/evaluate_benchmark_metrics.py
```

### 7. Large-scale Analysis

```bash
jupyter lab scripts/report_figures.ipynb
```

---

## Reproducibility Indicators

The framework evaluates the following reproducibility-related reporting indicators:

- Dataset information
- Data splits
- Hyperparameters
- Hardware details
- Code repositories
- Pseudocode
- Theorems
- Proofs
- Method descriptions

---

## Benchmark Results

The pipeline variants were evaluated on a manually annotated benchmark of 60 papers.

| Pipeline | Accuracy | Cohen's Kappa |
| --- | ---: | ---: |
| PyMuPDF Pipeline (Llama 8b) | 0.5852 | 0.2456 |
| PyMuPDF Pipeline (Llama 70b) | 0.6963 | 0.4305 |
| Smart PyMuPDF Pipeline (Llama 8b) | 0.8444 | 0.6788 |
| Smart PyMuPDF Pipeline (Llama 70b) | 0.9056 | 0.8082 |
| Smart GROBID Pipeline (Llama 8b) | 0.8370 | 0.6652 |
| Smart GROBID Pipeline (Llama 70b) | 0.9200 | 0.8394 |

The strongest performance was achieved using the Smart GROBID pipeline with the larger Llama model.

---

## Large-Scale Analysis

Following benchmark evaluation, the Smart GROBID pipeline was applied to more than 3,000 papers from AAAI, ACL, and NeurIPS.

The analysis revealed:

- higher average reproducibility reporting scores for ACL and NeurIPS compared to AAAI,
- increasing reporting scores over time,
- shifts following checklist adoption,
- and differing code/data reporting patterns across venues.

<p align="center">
  <img src="figures/final_venue_mean_repro_score_2023_2025.png" alt="Mean reproducibility score by conference" width="700">
</p>

---

## Notes

- The repository does not include the full paper corpus due to size limitations.
- Exact outputs may vary across models, providers, and reruns.
- The scraper used during this project was:
  https://github.com/george-gca/ai_papers_scrapper
```
