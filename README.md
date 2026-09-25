# Amazon ML Challenge 2026: Business Entity Resolution Pipeline

High-performance, fully vectorized, modular Business Entity Resolution engine designed for large-scale multi-source record linkage optimizing for Macro-Averaged $F_{0.5}$ (singleton-inclusive).

---

## ⚡ Quickstart (Clone & Run in 60 Seconds)

### 1. Environment Setup
```bash
# 1. Create and activate a Python virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.venv\Scripts\activate          # On Windows PowerShell

# 2. Install pinned dependencies
pip install -r requirements.txt
```

### 2. Verify Your Environment (Instant Smoke Test)
Run unit tests to verify the end-to-end pipeline works on synthetic data in under 2 seconds:
```bash
pytest tests/ -v
```

### 3. Run Local Cross-Validation (Sample or Full Dataset)
```bash
# Fast iteration on a 25,000 entity sample (takes ~30-60 seconds)
python run_pipeline.py --mode cv --sample-size 25000

# Full validation on 100% of the training dataset
python run_pipeline.py --mode cv --full
```

### 4. Generate Final Test Submission
```bash
# Runs full inference on test_source1, test_source2, test_source3
python run_pipeline.py --mode submit
# Output will be written to: submission/submission.csv
```

---

## 🏗️ Architecture & Modular Codebase

This repository is built with **strict modular interfaces** so developers can experiment in complete isolation without breaking downstream components.

```
amazon-ml-2026/
├── run_pipeline.py             # Master CLI runner (CV, Training, Inference)
├── requirements.txt            # Pinned high-performance dependencies
├── src/
│   ├── config.py               # Central paths, hyperparams, column mappings
│   ├── preprocessing/          # Vectorized text/phone/address normalization
│   │   └── cleaner.py
│   ├── blocking/               # TF-IDF cosine + exact hash candidate blocker
│   │   └── blocker.py
│   ├── features/               # Pairwise string, phonetic, and geo features
│   │   └── extractor.py
│   ├── models/                 # GroupKFold LightGBM pairwise ranker
│   │   └── ranker.py
│   ├── decision/               # Expected-F_0.5 dynamic threshold & singleton optimizer
│   │   └── optimizer.py
│   ├── evaluation/             # Official Macro F_0.5 metric implementation
│   │   └── metrics.py
│   └── pipeline.py             # Orchestrates the 5-stage pipeline
└── tests/
    └── test_pipeline.py        # End-to-end unit test suite
```

---

## 👥 Independent Development & Experimentation Guide

Any developer can modify individual components independently:

### 🔹 Experimenting with Preprocessing (`src/preprocessing/cleaner.py`)
- Add custom token replacements, regex rules for international corporate formats, or address parsing.
- Contract: Takes a Polars DataFrame $\rightarrow$ Returns a Polars DataFrame with `clean_*` columns.

### 🔹 Experimenting with Blocking (`src/blocking/blocker.py`)
- Try FAISS dense embeddings, BM25, or alternative Q-gram sizes.
- Evaluate candidate recall directly with `MultiPassBlocker.evaluate_blocking_recall()`.
- Contract: Takes cleaned S1 and Satellite tables $\rightarrow$ Returns `pairs_df` with `[s1_id, candidate_id]`.

### 🔹 Experimenting with Features (`src/features/extractor.py`)
- Add phonetic similarity (Soundex/Metaphone), geographical distances, or deep token overlap metrics.
- Contract: Takes candidate pairs $\rightarrow$ Returns features DataFrame with prefix `feat_*`.

### 🔹 Experimenting with Models (`src/models/ranker.py`)
- Tune hyperparameters, test CatBoost / XGBoost, or build ranker ensembles.
- Contract: Trains via GroupKFold on `s1_id` $\rightarrow$ Outputs out-of-fold match probabilities.

### 🔹 Tuning Decision Thresholds (`src/decision/optimizer.py`)
- Tune singleton gating logic ($\tau_{single}$) and match selection thresholds ($\tau_{match}$) to maximize Macro $F_{0.5}$.

---

## 🔒 Data Privacy & Git Safety
The `.gitignore` is pre-configured to **strictly exclude**:
- Raw datasets (`6ab10eb3b23ba_student_resource/`, `data/`, `*.csv`, `*.parquet`)
- Python caches (`__pycache__/`, `.venv/`)
- Model binaries & prediction outputs

Your gigabyte datasets will never be accidentally committed to GitHub.
