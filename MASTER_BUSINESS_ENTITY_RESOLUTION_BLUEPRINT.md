# MASTER BUSINESS ENTITY RESOLUTION BLUEPRINT
## Amazon ML Challenge 2026 — End-to-End System Architecture & High-Performance Execution Guide

**Metric:** Macro-averaged $F_{0.5}$ per Source-1 entity (singleton-inclusive, precision weighted $2\times$ over recall).  
**Rules & Constraints:** Final model $\le 8\text{B}$ parameters, MIT/Apache 2.0 License, strictly self-contained (zero external APIs, gazetteers, or web lookups).

---

## 1. Verified Data Audit & Empirical Baseline

Based on direct inspection of the 24.2 million total records across training and test datasets:

### 1.1 Dataset Statistics
| Dataset | Source | Total Rows | Country Distribution |
| :--- | :--- | :--- | :--- |
| **Train** | `train_source1.tsv` | **2,206,821** | US: 1,323,633 (60.0%), India: 883,188 (40.0%) |
| **Train** | `train_source2.tsv` | **5,034,616** | US: 3,016,817 (59.9%), India: 2,017,799 (40.1%) |
| **Train** | `train_source3.tsv` | **5,285,603** | US: 3,170,056 (60.0%), India: 2,115,547 (40.0%) |
| **Train** | `train_ground_truth.tsv` | **2,206,821** | Singletons: **123,247** (5.58%), Exploded Pairs: **7,638,365** |
| **Test** | `test_source1.tsv` | **1,732,544** | India: 809,986 (46.8%), US: 663,106 (38.3%), France: 259,452 (15.0%) |
| **Test** | `test_source2.tsv` | **4,887,273** | India: 2,312,565 (47.3%), US: 1,871,330 (38.3%), France: 703,378 (14.4%) |
| **Test** | `test_source3.tsv` | **5,082,316** | India: 2,405,000 (47.3%), US: 1,945,701 (38.3%), France: 731,615 (14.4%) |

### 1.2 Ground Truth Structural Properties
1. **Asymmetric 1-to-Many Mapping**: Each non-singleton S1 entity links to an average of **3.67** records (Median: **4.0**, 75th percentile: **5.0**, Max: **11**).
2. **Match Split**: S2 provides **48.36%** of matches; S3 provides **51.64%** of matches.
3. **Partition Cleanliness**: Exactly **0** matched S2/S3 IDs map to multiple S1 entities. Ground truth is a strict disjoint partition.
4. **Singleton Valuation**: Singletons account for **5.58%** of S1. Correctly predicting an empty list gives $F_{0.5} = 1.0$. A single false positive drops that entity to $0.0$.
5. **Open-Set Generalization**: France accounts for **15.0%** of test records and $0\%$ of training records. **No feature or blocking step can hard-filter on country.**

---

## 2. High-Throughput System Architecture

Due to the dataset scale (~12.5M train records and ~11.7M test records), pure Python loops or single-threaded string comparisons are computationally infeasible. The system is designed with vectorized C++ acceleration (`rapidfuzz`, `sparse_dot_topn`, `scikit-learn`, `polars`/`pyarrow`, and `faiss`/`sentence-transformers`).

```mermaid
flowchart TD
    subgraph S1 [1. High-Performance Preprocessing]
        A[Raw TSVs S1, S2, S3] --> B[SIMD Normalization & Cleaning]
        B --> C[Fast Romanization & Script Transliteration: anyascii]
        B --> D[Numeric Signature Digit-Run Extraction: \d+]
        B --> E[Corpus Legal-Suffix & Abbreviation Miner]
    end

    subgraph S2 [2. Multi-Channel Candidate Generation]
        C & D & E --> F1[Channel A: Exact / Canonical Name Hash]
        C & D & E --> F2[Channel B: Char 3-Gram TF-IDF Sparse Cosine]
        C & D & E --> F3[Channel C: Token Set Overlap Inverted Index]
        C & D & E --> F4[Channel D: Dense Multilingual Embedding ANN]
        C & D & E --> F5[Channel E: Numeric Signature Address Inverted Index]
        F1 & F2 & F3 & F4 & F5 --> G[Union & Provenance Tracking]
        G --> H[Intra-Source Deduplication & Cluster Merging S2-S2, S3-S3]
    end

    subgraph S3 [3. Stage-1 Pruning & Feature Pipeline]
        H --> I[Stage-1 Cheap Prune: Top 15-20 candidates/S1]
        I --> J[Recall@K Validation Gate >= 97%]
        J --> K[Full 35+ Feature Vector Computation on Survivors]
        K --> L[Group-Relative / Listwise Z-Scores & Score Gaps]
    end

    subgraph S4 [4. Two-Head Calibrated Modeling & Decision Layer]
        K & L --> M[Stage-2 LightGBM Classifier]
        M --> N[Per-Source Isotonic Calibration S2 vs S3]
        N --> O[Head 1: Singleton Gate Classifier]
        N --> P[Head 2: Dynamic Expected-F0.5 Selector]
        O & P --> Q[Final Multi-Accept Entity Cluster Engine]
    end

    subgraph S5 [5. Output Generation & Format Verification]
        Q --> R[matching_results.tsv]
        I --> S[candidate_pairs.tsv]
        R & S --> T[validate_submission.py Verification PASS]
    end
```

---

## 3. Data Preprocessing & Linguistic Normalization

### 3.1 Text Cleaning Engine (`normalize_name.py`, `normalize_address.py`)
- **Symbol & Prefix Stripping**: Remove regex `^[-\*#@\s]+` (e.g., `--Holloway Peak Inc` $\rightarrow$ `Holloway Peak Inc`).
- **URL Dissection**: Detect domain patterns (`.com`, `.org`, `.net`, `.in`, `.co`, `.io`). Strip protocol/TLDs and split remaining tokens on `.` and `-` (e.g., `agrosheronhotels.com` $\rightarrow$ `agro sheron hotels`).
- **DBA & Trade Names**: Split on ` dba `, ` d/b/a `, ` t/a `, ` trading as ` into primary and secondary entity aliases.
- **Literal Null Handling**: Convert `"null"`, `"NULL"`, `"(null)"`, `"none"`, `"-"` into empty strings.
- **Token De-duplication**: Replace repeated adjacent tokens (e.g., `Crestline Crestline Clean LP` $\rightarrow$ `Crestline Clean LP`).

### 3.2 High-Speed Script Transliteration (`transliterate.py`)
- **Challenge**: Multilingual Indian scripts (Devanagari, Tamil, Telugu, Gujarati, Bengali, Malayalam, Kannada) appear in S2/S3, while S1 is Latin.
- **Solution**: High-throughput romanization via `anyascii` (C-optimized, zero external API, handles all Indic scripts + French diacritics).
- **Dual Representation**: Store both raw text and romanized text for every record to enable dual-channel lexical + phonetic matching.

### 3.3 Numeric Signature Extraction (`numeric_signature.py`)
- Extract all digit sequences `\d+` from `business_address`.
- Store as sorted integer sets (e.g., `"H.No. 16-11-23/37/A, PIN 500036"` $\rightarrow$ `{"16", "11", "23", "37", "500036"}`).
- Provides an invariant address fingerprint resistant to word reordering and missing street names.

---

## 4. Multi-Channel Candidate Generation (Blocking)

To achieve $\ge 97\%$ Recall@K on 1.73M S1 entities while keeping candidate pool $\le 15$ per entity:

### 4.1 Blocking Strategies
1. **Channel A (Exact & Standardized Key)**: Exact match on normalized name root + country bucket (with cross-country fallback).
2. **Channel B (Character 3-Gram Sparse TF-IDF)**: 
   - Fit `TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=2)` on combined names and addresses.
   - High-speed sparse matrix multiplication using `sparse_dot_topn` or Scipy sparse matrix slicing to retrieve top-K nearest neighbors.
3. **Channel C (Token Inverted Index)**:
   - Index distinctive name tokens (corpus frequency $< 0.005$) and match records sharing $\ge 2$ rare tokens.
4. **Channel D (Dense Multilingual Embeddings)**:
   - Model: `intfloat/multilingual-e5-small` or `paraphrase-multilingual-mpnet-base-v2` ($\le 120\text{M}$ params, Apache 2.0).
   - Encode romanized names + addresses; index via FAISS (`IndexFlatIP` / `IndexIVFFlat`).
5. **Channel E (Address Numeric Signature Index)**:
   - Index records sharing rare digit patterns ($\ge 4$ digits or PIN/postal code matches).

### 4.2 Intra-Source Clustering (`intra_source_clustering.py`)
- Run near-duplicate blocking within S2 alone and S3 alone.
- Group internal duplicates into connected components ($> 0.90$ similarity).
- When any record in an intra-source cluster is retrieved for an S1 entity, add its cluster siblings.

---

## 5. Feature Engineering Battery (35+ Features)

Computed on Stage-1 survivors using SIMD-accelerated `rapidfuzz`:

### 5.1 Name Similarity Features
1. `name_levenshtein_ratio` (Normalized Levenshtein)
2. `name_jaro_winkler` (Prefix-weighted similarity)
3. `name_token_sort_ratio` (Handles reordered tokens)
4. `name_token_set_ratio` (Handles subset/duplicated tokens)
5. `name_char_3gram_cosine` (Sub-token overlap)
6. `name_length_ratio` & `name_length_diff`
7. `name_token_count_diff`
8. `name_first_token_match` (Binary flag)
9. `name_numeric_token_match` (Catches OCR `0` vs `o`, `1` vs `l`)
10. `name_soft_tfidf` (Corpus-IDF weighted Monge-Elkan token similarity)
11. `name_embedding_cosine` (Multilingual embedding dot-product)
12. `romanized_name_similarity`

### 5.2 Address Similarity Features
13. `address_token_jaccard`
14. `address_char_3gram_cosine`
15. `address_numeric_jaccard` (Jaccard similarity of digit-run sets)
16. `address_numeric_overlap_count`
17. `postal_code_exact_match` (Binary flag: 1 if matching, 0 if mismatch, -1 if missing)
18. `address_missing_s1` & `address_missing_candidate` (Missingness indicators)
19. `address_embedding_cosine`

### 5.3 Cross-Field & Retrieval Provenance Features
20. `country_exact_match` (Soft feature: 1 for match, 0 for mismatch)
21. `name_x_address_sim` (Interaction product term)
22. `retrieval_channel_count` (Number of blocking channels that retrieved this pair: 1 to 5)
23. `retrieval_min_rank` (Best rank across all retrieval channels)
24. `source_origin_s2` & `source_origin_s3` (One-hot candidate source)

### 5.4 Listwise & Group-Relative Features (Calculated per S1)
25. `name_sim_zscore` (Z-score of name similarity relative to all candidates for this S1)
26. `address_sim_zscore`
27. `combined_score_gap_to_next` (Difference between this candidate and the next best candidate)
28. `combined_score_percentile`
29. `candidate_pool_size` (Total candidates retrieved for this S1)

---

## 6. Two-Head Calibrated Classifier & Metric Optimization

### 6.1 LightGBM Pairwise Classifier (`pairwise_gbm.py`)
- Loss: Binary Logloss / Focal Loss.
- Grouping: `GroupKFold` split strictly by `source1_entity_id` (zero data leakage).
- Hyperparameters tuned via Optuna on validation Macro-$F_{0.5}$.

### 6.2 Per-Source Isotonic Calibration (`calibration.py`)
- S2 and S3 have distinct noise characteristics. Fit separate isotonic regressors $C_{S2}(s)$ and $C_{S3}(s)$ to map raw model scores into true posterior probabilities $P(\text{Match} \mid \text{Score})$.

### 6.3 Dedicated Singleton Gate (`singleton_gate.py`)
- Head 1 Classifier: Given S1-level aggregate features:
  $$\vec{x}_{S1} = [\max_i(p_i), \text{mean}(p_i), \text{std}(p_i), \text{entropy}(p), \text{candidate\_count}]$$
- Predicts $P(\text{Singleton} \mid S1)$. If $P(\text{Singleton}) > \tau_{\text{singleton}}$, immediately output empty match set `""` (earning $F_{0.5} = 1.0$).

### 6.4 Expected-$F_{0.5}$ Dynamic Accept-Set Selector (`expected_f05_selector.py`)
For non-singletons with calibrated probabilities $p_1 \ge p_2 \ge \dots \ge p_m$:
- For each candidate accept subset of size $k \in \{1, \dots, m\}$:
  $$\mathbb{E}[\text{Precision}_k] = \frac{1}{k} \sum_{i=1}^k p_i$$
  $$\mathbb{E}[\text{Recall}_k] = \frac{\sum_{i=1}^k p_i}{\sum_{j=1}^m p_j + \epsilon}$$
  $$\mathbb{E}[F_{0.5}(k)] = \frac{1.25 \times \mathbb{E}[\text{Precision}_k] \times \mathbb{E}[\text{Recall}_k]}{0.25 \times \mathbb{E}[\text{Precision}_k] + \mathbb{E}[\text{Recall}_k]}$$
- Select $k^* = \arg\max_k \mathbb{E}[F_{0.5}(k)]$.
- Guaranteed to directly maximize the challenge evaluation metric.

---

## 7. Submission Package Compliance & Layout

The submission zip structure matches the exact competition specification:

```
submission.zip
├── output/
│   ├── matching_results.tsv        # Scored leaderboard file (source1_entity_id \t matched_entity_ids)
│   └── candidate_pairs.tsv         # Blocking candidate set (source1_entity_id \t candidate_entity_ids)
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── preprocessing/
│       │   │   ├── normalize.py
│       │   │   ├── transliterate.py
│       │   │   └── numeric_signature.py
│       │   ├── blocking/
│       │   │   ├── blocking_tfidf.py
│       │   │   ├── blocking_embedding.py
│       │   │   ├── intra_source_clustering.py
│       │   │   └── candidate_union.py
│       │   ├── features/
│       │   │   ├── similarity_features.py
│       │   │   ├── soft_tfidf.py
│       │   │   └── listwise.py
│       │   ├── models/
│       │   │   ├── train_gbm.py
│       │   │   ├── calibrate.py
│       │   │   └── singleton_gate.py
│       │   ├── decision/
│       │   │   └── expected_f05_selector.py
│       │   └── pipeline.py         # One-click end-to-end execution
│       ├── README.md               # Exact reproduction commands
│       └── requirements.txt        # Pinned dependencies
└── Documentation_template.md       # Complete technical methodology report
```

### Automatic Pre-Submission Gate
Before final submission, the pipeline runs:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```
Ensures 100% compliance with tab delimiters, entity ID completeness (1,732,544 rows), singleton representation, and S2/S3 prefix formatting.

---

## 8. Step-by-Step Implementation Roadmap

```
[Phase 1: Foundation & Preprocessing]
  ├── Step 1.1: Build fast vectorizer & cleaner (normalize_name, normalize_address)
  ├── Step 1.2: Build anyascii romanizer + numeric signature extraction
  └── Step 1.3: Generate processed/normalized train & test caches

[Phase 2: High-Recall Multi-Channel Blocking]
  ├── Step 2.1: TF-IDF sparse matrix top-K blocking
  ├── Step 2.2: Intra-source clustering (S2-S2, S3-S3)
  ├── Step 2.3: Dense embedding retrieval channel
  └── Step 2.4: Union candidates & verify Recall@K >= 97% on validation split

[Phase 3: Vectorized Feature Engine]
  ├── Step 3.1: Rapidfuzz SIMD similarity suite
  ├── Step 3.2: Numeric signature Jaccard & Soft-TF-IDF
  └── Step 3.3: Group-relative listwise features (z-scores, score gaps)

[Phase 4: Model Training, Calibration & Expected-F0.5 Decision]
  ├── Step 4.1: Train LightGBM with GroupKFold by S1
  ├── Step 4.2: Fit per-source Isotonic calibration
  ├── Step 4.3: Train Singleton Gate classifier
  └── Step 4.4: Dynamic Expected-F0.5 optimizer

[Phase 5: Inference, Validation & Submission Packaging]
  ├── Step 5.1: Run full pipeline on test dataset (1.73M S1 entities)
  ├── Step 5.2: Output matching_results.tsv & candidate_pairs.tsv
  ├── Step 5.3: Run validate_submission.py
  └── Step 5.4: Package submission.zip with code & Documentation_template.md
```
