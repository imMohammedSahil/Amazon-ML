# Amazon ML Challenge 2026 — Business Entity Resolution

**Task:** Large-scale multi-source record linkage across three heterogeneous business entity datasets.  
**Metric:** Macro-Averaged $F_{0.5}$ (precision-weighted) over 1,732,544 Source 1 anchor entities.  
**Public Leaderboard Score:** `0.762`  
**Offline Benchmark (holdout):** `0.7779` — Precision `84.04%`, Recall `68.49%`

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [End-to-End Pipeline Architecture](#2-end-to-end-pipeline-architecture)
3. [Multi-Pass Candidate Blocking](#3-multi-pass-candidate-blocking)
4. [Pairwise Feature Engineering](#4-pairwise-feature-engineering)
5. [Ensemble Model Architecture](#5-ensemble-model-architecture)
6. [Decision Layer and Singleton Gating](#6-decision-layer-and-singleton-gating)
7. [Evaluation Metric — Macro F0.5](#7-evaluation-metric--macro-f05)
8. [Precision vs. Recall Trade-off Analysis](#8-precision-vs-recall-trade-off-analysis)
9. [Module Interface Contracts](#9-module-interface-contracts)
10. [Directory Structure](#10-directory-structure)
11. [Setup and Execution](#11-setup-and-execution)

---

## 1. Problem Formulation

Three independent sources (S1, S2, S3) each hold partial, noisy fragments of information about the same real-world business entities. The objective is to resolve which S2/S3 records refer to the same physical entity as each S1 anchor record.

```
Source 1 (Reference/Anchors)  ->  N = 1,732,544 entities
Source 2 (Satellite A)        ->  independent business registry
Source 3 (Satellite B)        ->  independent business registry
```

Key challenges:

- **Scale:** O(N x M) pairwise comparison is intractable. Blocking is mandatory and graded as a separate deliverable.
- **Noise:** Name transliterations (Latin vs. Indic scripts), legal suffix variants, address abbreviation heterogeneity.
- **Singletons:** A significant fraction of S1 anchors have no S2/S3 match. Correctly predicting an empty set scores `1.0` for that entity. Incorrectly predicting non-empty scores `0.0`.
- **Multilingual:** Records span French, English, Hindi transliterations, and other regional scripts.

---

## 2. End-to-End Pipeline Architecture

```
Raw Input (S1, S2, S3 TSV files)
         |
         v
+---------------------------------------------+
|  Stage 1 -- Preprocessing & Normalization   |
|  - Unicode normalization, accent stripping  |
|  - Legal suffix stripping                   |
|    (Inc, LLC, SARL, GmbH, SAS, EURL, ...)  |
|  - Leet-speak normalization (0->o, 1->l)   |
|  - Address token standardization            |
|    (Rue/R, Ave/Avenue, Blvd/Boulevard, ...) |
|  - Country code canonicalization            |
|    (USA / United States -> "us")            |
|  - Phonetic features: Soundex, consonant    |
|    skeleton, sorted 3-gram signatures       |
+-------------------+-------------------------+
                    |
                    v
+---------------------------------------------+
|  Stage 2 -- 23-Pass Rule-Based Blocking     |
|  Reduces O(N x M) -> O(K x N), K <= 35     |
|  Each pass is a deterministic equi-join     |
|  on composite blocking keys (see Section 3) |
+-------------------+-------------------------+
                    |
                    v
+---------------------------------------------+
|  Stage 3 -- Pairwise Feature Extraction     |
|  40 engineered features per candidate pair  |
|  (string distances, phonetic matches,       |
|  address overlap, digit equality)           |
+-------------------+-------------------------+
                    |
                    v
+---------------------------------------------+
|  Stage 4 -- 5-Seed LightGBM Ensemble       |
|  Binary classification P(match | s1, cand)  |
|  Weighted average of 5 independently seeded |
|  models with varied depths and leaves       |
|  Cascaded exact-match probability boosts    |
|  Adjacent door-number penalty dampening     |
+-------------------+-------------------------+
                    |
                    v
+---------------------------------------------+
|  Stage 5 -- Decision Layer                  |
|  Singleton gating via probability threshold |
|  Multi-source transitivity booster (1.015x) |
|  Injective 1-to-1 mapping enforcement       |
+-------------------+-------------------------+
                    |
                    v
         matching_results.tsv
         candidate_pairs.tsv
```

---

## 3. Multi-Pass Candidate Blocking

Blocking reduces the search space from a full cross-product to a tractable candidate set. All 23 passes run as exact equi-joins on composite keys; the union is deduplicated per `(s1_id, cand_id)`. Country-scoped joins prevent cross-region false positives.

| Pass | Category | Key A | Key B | Key C | Top-K | Rationale |
|:----:|:---------|:------|:------|:------|:-----:|:----------|
| 1 | Name Exact | `name` (exact) | `country_std` | — | 4 | Catches identical cleaned names in the same country |
| 2 | Name Stem | `name_root` | `country_std` | — | 4 | Legal-suffix-stripped root match within country |
| 3 | Name Token-Sort | `sorted_root` | `country_std` | — | 3 | Word-order-agnostic root match (e.g., "Ali Bakery" vs "Bakery Ali") |
| 4 | Address + Name | `street_digits` | `root_prefix_3` | `country_std` | 4 | Same door number + name prefix — strong address anchor |
| 5 | Unit + Name | `unit_code` | `root_prefix_3` | `country_std` | 3 | Suite/shop number with name prefix |
| 6 | PIN + Name | `postal_code` | `root_prefix_3` | `country_std` | 4 | Postal code + name prefix — broad area anchor |
| 7 | PIN + Street | `postal_code` | `street_digits` | `country_std` | 3 | Same postcode AND door number — transliteration-robust |
| 8 | Phonetic + PIN | `soundex_first` | `postal_code` | `country_std` | 4 | Phonetically similar first word in same postal area |
| 9 | Phonetic + Street | `soundex_first` | `street_digits` | `country_std` | 4 | Phonetically similar first word at same door number |
| 10 | Consonant + PIN | `consonant_stem` | `postal_code` | `country_std` | 3 | Consonant skeleton + postcode (script-independent) |
| 11 | Consonant + Street | `consonant_stem` | `street_digits` | `country_std` | 3 | Consonant skeleton + door number |
| 12 | Acronym + Street | `acronym` | `street_digits` | `country_std` | 3 | Abbreviated name vs. satellite full name at same address |
| 13 | Phonetic2 + PIN | `soundex_second` | `postal_code` | `country_std` | 3 | Phonetic match on second name word + postcode |
| 14 | Phonetic2 + Street | `soundex_second` | `street_digits` | `country_std` | 3 | Phonetic match on second name word + door number |
| 15 | Addr Prefix + PIN | `addr_core_prefix_4` | `postal_code` | `country_std` | 3 | First 4 chars of stripped address + postcode |
| 16 | 3-Key Address | `root_prefix_2` | `street_digits` | `postal_code` | 3 | Triple-anchor: name prefix + door + postcode (high precision) |
| 17 | 3-Gram + PIN | `sorted_3gram` | `postal_code` | `country_std` | 3 | Sorted character 3-gram signature + postcode |
| 18 | Unit + PIN | `unit_code` | `postal_code` | `country_std` | 3 | Suite number + postal code match |
| 19 | Addr Core Exact | `addr_core` (≥8 chars) | `country_std` | — | 3 | Long stripped-address exact match within country |
| 20 | Phonetic + Addr | `soundex_first` | `addr_core_prefix_4` | `country_std` | 3 | Phonetic name + address prefix — catches transliterations |
| 21 | Name + PIN (global) | `name_root` (≥5 chars) | `postal_code` | — | 2 | Country-agnostic fallback: long root + postcode |
| 22 | Consonant + Addr | `consonant_stem` | `addr_core_prefix_4` | `country_std` | 3 | Consonant skeleton + address prefix |
| 23 | First Word + Street | `first_word` (≥4 chars) | `street_digits` | `country_std` | 3 | First meaningful name word + door number |

**Derived blocking keys:**

| Key | Source Field | Derivation |
|:----|:-------------|:-----------|
| `name_root` | `business_name` | Strip legal prefixes/suffixes (Inc, LLC, SARL, …), leet-normalize, lowercase |
| `sorted_root` | `name_root` | Token-sort words of `name_root` alphabetically and rejoin |
| `acronym` | `name_root` | Concatenate first character of each `name_root` token |
| `first_word` | `name_root` | First token of `name_root` |
| `soundex_first` | `first_word` | Soundex code (4-char phonetic encoding) |
| `soundex_second` | second token of `name_root` | Soundex code of second word |
| `consonant_stem` | `first_word` | First 3 consonants of first word (script-independent skeleton) |
| `sorted_3gram` | `name_root` | Alphabetically sorted characters of first 4-char prefix |
| `root_prefix_N` | `name_root` | First N characters, N ∈ {2, 3, 4, 5} |
| `street_digits` | `business_address` | Leading digit sequence (door/building number) |
| `postal_code` | `business_address` | 5–6 digit postal/PIN code extracted via regex |
| `unit_code` | `business_address` | Suite/unit/flat number extracted via regex |
| `addr_core` | `business_address` | Address with street-type tokens stripped (rue, ave, blvd, …) |
| `addr_core_prefix_4` | `addr_core` | First 4 characters of `addr_core` |
| `country_std` | `country` | Canonicalized country code (USA → us, United Kingdom → gb, …) |



---



## 4. Pairwise Feature Engineering

Each candidate pair `(s1_id, cand_id)` is featurized into a 40-dimensional vector computed in a single NumPy pass for throughput:

| Index | Feature | Description |
|:------|:--------|:------------|
| 0 | `exact_name` | Exact cleaned name match |
| 1 | `exact_root` | Exact name_root match |
| 2 | `jw_root` | Jaro-Winkler similarity on name_root |
| 3 | `sort_root` | Token sort ratio on name_root |
| 4 | `set_root` | Token set ratio on name_root |
| 5 | `partial_root` | Partial string ratio on name_root |
| 6 | `lev_root` | Normalized Levenshtein similarity on name_root |
| 7 | `qratio_root` | Quick ratio on name_root |
| 8 | `len_diff_ratio` | Absolute length difference / max length |
| 9 | `acronym_match` | Acronym vs. root or acronym vs. acronym equality |
| 10 | `soundex_match` | Soundex code equality (first word) |
| 11 | `consonant_match` | Consonant skeleton equality (>=2 consonants) |
| 12 | `jw_soundex` | Jaro-Winkler on Soundex codes |
| 13 | `jw_addr` | Jaro-Winkler on standardized addresses |
| 14 | `sort_addr` | Token sort ratio on standardized addresses |
| 15 | `set_addr` | Token set ratio on standardized addresses |
| 16 | `lev_addr` | Normalized Levenshtein on addresses |
| 17 | `core_addr_sort` | Token sort on addr_core |
| 18 | `core_addr_set` | Token set on addr_core |
| 19 | `core_addr_lev` | Levenshtein on addr_core |
| 20 | `same_digits` | street_digits equality indicator |
| 21 | `diff_digits` | street_digits inequality indicator |
| 22 | `digit_num_diff` | Numeric door-number delta, clipped to 50 |
| 23 | `same_unit` | unit_code equality |
| 24 | `same_pin` | postal_code equality |
| 25 | `diff_pin` | postal_code inequality |
| 26 | `same_country` | country_std equality |
| 27 | `translit_flag` | sort_addr >= 0.70 AND same_digits=1 (transliteration signal) |
| 28 | `composite` | max(jw_root, sort_root, acronym_match) x sort_addr |
| 29 | `addr_match_score` | 0.7 x sort_addr + 0.3 x same_digits |
| 30 | `pin_and_digits` | same_pin=1 AND same_digits=1 |
| 31 | `phonetic_addr_composite` | max(soundex_match, consonant_match) x sort_addr |
| 32 | `strong_exact_pair` | exact_root=1 AND (same_digits=1 OR same_pin=1) |
| 33 | `root_contains` | Substring containment between roots (>=4 chars each) |
| 34 | `exact_full_addr` | Full standardized address exact match |
| 35 | `exact_root_and_pin` | exact_root=1 AND same_pin=1 |
| 36 | `exact_addr_core` | addr_core exact match (>=6 chars) |
| 37 | `shared_tokens` | Token Jaccard: |tokens1 & tokens2| / |tokens1 | tokens2| |
| 38 | `adjacent_door_penalty` | Numeric door delta in [1, 4] -- near-miss penalty |
| 39 | `sort_set_diff` | |sort_root - set_root| -- disambiguation feature |

---

## 5. Ensemble Model Architecture

Five LightGBM binary classifiers are trained independently with varied random seeds, tree depths, and regularization hyperparameters. Predictions are combined via a fixed weighted average:

$$P_{\text{final}} = 0.25 \cdot P_1 + 0.25 \cdot P_2 + 0.20 \cdot P_3 + 0.15 \cdot P_4 + 0.15 \cdot P_5$$

**Model configurations:**

| Model | Weight | Seed | LR | Leaves | Depth | pos\_weight | reg\_alpha | reg\_lambda | Rounds | Role |
|:------|:------:|-----:|---:|-------:|------:|------------:|-----------:|------------:|-------:|:-----|
| M1 | 0.25 | 42 | 0.050 | 85 | 9 | 1.30 | 0.05 | 0.5 | 200 | Primary anchor — balanced generalization |
| M2 | 0.25 | 1337 | 0.045 | 95 | 10 | 1.25 | 0.10 | 0.8 | 220 | Primary anchor — higher L1/L2 regularization |
| M3 | 0.20 | 2026 | 0.055 | 75 | 8 | 1.35 | 0.08 | 0.6 | 180 | Shallower trees — reduces overfitting on noisy pairs |
| M4 | 0.15 | 777 | 0.040 | 105 | 10 | 1.20 | 0.12 | 0.9 | 240 | Wider leaves — captures complex address interactions |
| M5 | 0.15 | 999 | 0.035 | 120 | 11 | 1.25 | 0.15 | 1.0 | 260 | Deepest/widest — diversity booster, high regularization |

- `pos_weight`: upweights positive (match) class to compensate for class imbalance  
- `reg_alpha` / `reg_lambda`: L1 / L2 regularization on leaf weights  
- Weights sum to 1.0; M1 and M2 carry equal highest weight as they demonstrated best individual CV scores

**Post-prediction probability corrections applied after ensemble averaging:**

| Rule | Condition | Effect |
|:-----|:----------|:-------|
| Exact-root boost | `exact_root=1` AND (`same_digits=1` OR `same_pin=1`) | `P_final = max(P_final, 0.998)` — near-certain match |
| Adjacent door penalty | `adjacent_door_penalty=1` AND `exact_root=0` | `P_final = min(P_final, 0.75)` — cap near-miss door numbers |
| Transitivity booster | Both S2 and S3 yield score ≥ 0.88 for same S1 | Multiply high-confidence S2 scores by 1.015x |


---

## 6. Decision Layer and Singleton Gating

```
For each S1 entity (s1_id):

  candidates <- {(cand_id, P_final) from S2 union S3 blocking}

  IF max(P_final) < PROB_CUTOFF (0.992):
      prediction = []           <- Singleton (no match)
  ELSE:
      threshold = max(P_final) x MARGIN_RATIO (0.88)
      selected  = {c : P_final(c) >= threshold}
      prediction = top-K(selected, K=3)
```

**Injective mapping:** A secondary pass enforces that no satellite ID is assigned to more than one S1 entity, resolving conflicts by highest probability score.

**Tuned hyperparameters (offline benchmark: Macro F0.5 = 0.7779):**

| Parameter | Value | Description |
|:----------|:------|:------------|
| `PROB_CUTOFF` | 0.992 | Minimum max-score to predict non-empty |
| `MARGIN_RATIO` | 0.88 | Fraction of max score for secondary candidates |
| `TOP_K` | 3 | Maximum candidates retained per S1 entity |
| `CHUNK_SIZE` | 200,000 | Streaming inference chunk (memory cap < 8.5 GB) |

---

## 7. Evaluation Metric — Macro F0.5

The competition metric is Macro-Averaged $F_{0.5}$ computed over all N anchor entities.

$$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

$\beta = 0.5$ means precision is weighted **4x more** than recall in the harmonic mean.

**Per-entity scoring rules:**

| Ground Truth $Y_i$ | Prediction $\hat{Y}_i$ | Score $S_i$ |
|:--------------------|:------------------------|:------------|
| $\emptyset$ (singleton) | $\emptyset$ | **1.0** |
| $\emptyset$ (singleton) | Non-empty | **0.0** |
| Non-empty | $\emptyset$ | **0.0** |
| Non-empty | Non-empty | $F_{0.5}(Y_i, \hat{Y}_i)$ |

$$\text{Macro } F_{0.5} = \frac{1}{N} \sum_{i=1}^{N} S_i$$

Singleton accuracy is critical: a false positive on a singleton entity contributes a full $-1/N$ penalty to the macro average.

---

## 8. Precision vs. Recall Trade-off Analysis

Two submission strategies were evaluated against the 50,000-entity holdout:

| Strategy | Macro F0.5 | Precision | Recall | Notes |
|:---------|:----------:|----------:|-------:|:------|
| LightGBM Ensemble (submitted) | **0.7779** | 84.04% | 68.49% | Final submission |
| Rule-Only Precision Maximizer | 0.3523 | 99.10% | 20.44% | Recall collapse — not submitted |

The rule-only 99%+ precision approach catastrophically collapsed recall. Despite $F_{0.5}$ precision weighting, the metric still requires meaningful recall to avoid a near-zero score. The LightGBM ensemble with `PROB_CUTOFF = 0.992` achieves the best empirical balance.

**Score sensitivity to PROB_CUTOFF threshold (offline holdout):**

| PROB\_CUTOFF | Macro F0.5 | Precision | Recall | Notes |
|:------------:|:----------:|----------:|-------:|:------|
| 0.990 | 0.7752 | 82.1% | 70.3% | |
| **0.992** | **0.7779** | **84.04%** | **68.49%** | Selected |
| 0.994 | 0.7703 | 86.5% | 65.1% | |
| 0.996 | 0.7580 | 89.2% | 60.4% | |


---

## 9. Module Interface Contracts

Each module in `src/` adheres to strict schema contracts to allow independent testing:

| Module | Path | Input | Output |
|:-------|:-----|:------|:-------|
| Preprocessing | `src/preprocessing/cleaner.py` | Raw Polars DataFrame (`id`, `business_name`, `business_address`, `country`) | DataFrame with `name_root`, `soundex_*`, `consonant_stem`, `postal_code`, `street_digits` columns |
| Blocking | `src/blocking/blocker.py` | Cleaned `df_s1`, `df_satellites` | DataFrame schema: `[s1_id, cand_id]` |
| Feature Extraction | `src/features/extractor.py` | Candidate pairs + cleaned source tables | NumPy array `(n_pairs, 40)` |
| Modeling | `src/models/ranker.py` | Feature matrix + labels | Match probabilities `P(match)` per pair |
| Decision Layer | `src/decision/optimizer.py` | Candidate probabilities + thresholds | `Dict[s1_id -> List[cand_id]]` |
| Evaluation | `src/evaluation/metrics.py` | Ground truth dict + prediction dict | `{macro_f05, precision, recall, singleton_accuracy}` |

---

## 10. Directory Structure

```
Amazon-ML/
|-- lgbm_grandmaster.py             # Production inference engine (5-seed ensemble, 23-pass blocking)
|-- run_pipeline.py                 # CLI orchestrator (cv / submit modes)
|-- requirements.txt                # Pinned dependency specification
|-- README.md
|
|-- src/
|   |-- config.py                   # Dataset paths, hyperparameter registry
|   |-- preprocessing/
|   |   `-- cleaner.py              # Vectorized Polars text normalization
|   |-- blocking/
|   |   `-- blocker.py              # Multi-pass equi-join candidate generator
|   |-- features/
|   |   `-- extractor.py            # 40-feature pairwise extractor (NumPy)
|   |-- models/
|   |   `-- ranker.py               # LightGBM binary classifier trainer
|   |-- decision/
|   |   `-- optimizer.py            # Singleton gating and threshold optimizer
|   |-- evaluation/
|   |   `-- metrics.py              # Competition-exact Macro F0.5 implementation
|   `-- pipeline.py                 # End-to-end pipeline orchestrator
|
|-- output/
|   |-- matching_results.tsv        # Final submission file (1,732,544 rows)
|   `-- candidate_pairs.tsv         # Blocking candidate output (graded separately)
|
|-- tests/
|   `-- test_pipeline.py            # Automated test suite
|
|-- aws_09_autonomous_champion.py   # Hyperparameter search / ablation runner (EC2)
|-- calibrate_precision_099.py      # Experimental high-precision submission generator
`-- 6ab10eb3b23ba_student_resource/ # Official competition dataset and validator
    `-- student_resource/
        |-- dataset/
        |   |-- train_source1.tsv
        |   |-- train_source2.tsv
        |   |-- train_source3.tsv
        |   |-- train_gt.tsv
        |   |-- test_source1.tsv
        |   |-- test_source2.tsv
        |   `-- test_source3.tsv
        `-- utils/
            `-- validate_submission.py
```

---

## 11. Setup and Execution

### Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows PowerShell

pip install -r requirements.txt
```

### Validate Submission Format

```bash
python 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test
```

### Run Local Offline Benchmark

```bash
# Cross-validation on training set (sampled)
python run_pipeline.py --mode cv --sample-size 25000

# Full training set validation
python run_pipeline.py --mode cv --full
```

### Generate Final Submission

```bash
python lgbm_grandmaster.py
```

Outputs written to `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### Run Test Suite

```bash
pytest tests/ -v
```

---

## Dependencies

| Package | Version | Role |
|:--------|:--------|:-----|
| `polars` | >= 1.0.0 | Columnar DataFrame operations (preprocessing, blocking) |
| `numpy` | >= 1.24.0 | Feature matrix construction |
| `lightgbm` | >= 4.0.0 | Gradient boosted binary classifier (5-model ensemble) |
| `rapidfuzz` | >= 3.0.0 | Jaro-Winkler, Levenshtein, token ratio computations |
| `scikit-learn` | >= 1.3.0 | GroupKFold cross-validation utilities |
| `scipy` | >= 1.10.0 | Sparse matrix operations |
| `pyarrow` | >= 12.0.0 | Parquet / Arrow I/O backend for Polars |
| `jellyfish` | >= 1.0.0 | Phonetic encoding (Soundex, NYSIIS) |

---

## Leaderboard

| Submission | Score | Timestamp | Status |
|:-----------|:------|:----------|:-------|
| LightGBM Ensemble (PROB_CUTOFF=0.992) | **0.762** | Sep 26, 2026 11:41 PM IST | Public |


