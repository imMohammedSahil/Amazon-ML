# Business Entity Resolution — Complete Implementation Blueprint

**Metric: F0.5 (β=0.5), macro-averaged per Source-1 entity, singleton-inclusive.**
**Constraint: final model ≤8B params, MIT/Apache license. No external lookups/APIs/gazetteers.**

This is the full, self-contained architecture document: the original problem/data audit, merged
with every enhancement identified in review. Nothing is appended at the end — each enhancement
is folded into the section it belongs to. Items are not separately tagged as "v1/v2"; this is the
one architecture to build.

---

## 1. What is actually being asked — and where the spec is genuinely ambiguous

**The task, precisely:** For every S1 record (deduplicated reference entities), find all S2 and
S3 records that refer to the same real-world business. Output is S1 → {0..n} S2/S3 ids. This is
**not classic symmetric pairwise dedup** — it's asymmetric: S1 is the clean anchor set, S2/S3 are
noisy satellite sources that plausibly contain their own internal duplicates.

**Structural fact from the ground-truth list shape:** match lists average ~4 ids, up to 7+, and
routinely contain **multiple S2 ids and multiple S3 ids for the same S1**. The most important
inference from this: **S2 and S3 are not internally deduplicated** — they likely contain repeat
listings of the same business — so the real task is closer to *recovering a whole cluster around
a fixed anchor* than scoring isolated pairs. This has direct architectural consequences (see §4,
intra-source clustering).

Roughly 6% of sampled GT rows are singletons (empty match list) — correctly predicting these
empty is worth a full 1.0 per the scoring rule, so singleton precision deserves dedicated
machinery, not just a low threshold (see §8).

**Explicit ambiguities to resolve against the real training files before building anything —
do not assume:**

1. **Is GT exhaustive?** If labelers missed true matches, your measured recall/precision will be
   biased pessimistic. Verify by manually auditing a handful of singleton S1 rows against S2/S3
   for suspiciously close unlabeled candidates.
2. **Can one S2/S3 id appear under more than one S1's match list?** Check with
   `groupby(matched_id).nunique(source1_entity_id)` on the exploded GT. This changes whether
   S2/S3 records need to be treated as potentially ambiguous between multiple real entities.
3. **Is an empty GT a verified true negative or an unlabeled gap?** The scoring rule (predicting
   empty on a true singleton = 1.0) implies GT singletons are meant to be trusted — but spot-check
   before fully relying on it.
4. **Country is an open set, not {US, India}.** France appears only in test. This rules out any
   hard country filter, one-hot encoding limited to two values, or country-indexed lookup table
   that doesn't degrade gracefully to an unseen label.

Before writing pipeline code: pull real GT rows and their corresponding S1/S2/S3 records from the
*same* file (not disjoint samples) and inspect them together — verify noise patterns, cardinality,
and the four ambiguities above against the actual data, not inference from samples.

---

## 2. Data profile

### Name noise (observed directly in the S2/S3 samples)

- **Legal-suffix variation:** Corp/Corporation, Ltd/Limited/LLP/Pvt/Private — appearing as
  leading *or* trailing tokens, sometimes bracketed: `[TECHNOLOGIES]`, `(LLC)`, `(Co)`, `[Group]`,
  `(India) Ltd`.
- **Character-level corruption / homoglyph noise:** `Nétwork`, `Cárolina`, `Cillgc`, `Medf0rd`
  (0 for o), `Léarning`, `Béque`, `Hospirlg` — looks like deliberately injected typo/OCR-style
  noise. Character-level methods (edit distance, n-gram) handle this far better than token-exact
  matching.
- **Duplicated tokens:** `Crestline Crestline Clean LP`, `Keystone Odyssey Odyssey LLC`,
  `TAMANNA and BROTHERS` — classic word-duplication injection.
- **Junk prefixes/symbols:** `--`, `***`, `#`, `@` prepended (`--Holloway Peak Inc`,
  `***Country Real`, `#centraleducation`, `@shreefoundation`). Strip in normalization; log
  frequency as a possible weak source-fingerprint feature.
- **Website-as-name:** `heassociates.com`, `prprivate.com`, `agrosheronhotels.com`,
  `Hmgreen.Com`, `wilfordhancock.com`, `lochravengolden.com` — a meaningful fraction of S2/S3
  names are literal URLs. Requires special tokenization (strip TLD, split on `.`/`-`) or these
  will look nothing like the matching S1 name.
- **DBA/trade names:** `Ectolumdrex dba X+ Madison Inc`, `Quonex d/b/a M D Herrera Offshore` —
  both the legal and DBA name may need independent matching against S1.
- **Multilingual names, not just addresses:** Devanagari, Tamil, Gujarati, Bengali, Kannada,
  Malayalam, Telugu scripts appear as the *business_name itself*
  (`राम मार्केटिंग प्राइवेट लिमिटेड`, `குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்`), and some names mix
  scripts within one string (`அரிஹந்த் Foundation Private Limited`). S1 in the sample is entirely
  Latin-script, even for Indian entities — cross-script S1↔S2/S3 name matching is a first-class,
  high-volume problem for the India subset, not an edge case.

### Address noise

- **Component order is inconsistent even within one source** — sometimes
  `city, state, street`, sometimes `street, city, state`, in the same file. Rules out any
  positional parsing; comparisons must be field-agnostic / bag-of-tokens.
- **Missing/blank addresses are common**, especially India-side S2/S3 records. A blank address
  must not zero out an otherwise strong name match — use missing-field indicator features, not
  silent zeros.
- **Literal `"NULL"`/`"null"` strings appear inline** (`"NULL, SPICEWOOD, TX"`) — normalize to
  true missing, don't treat as a token.
- **State field sometimes native-script** even when the rest of the record is Latin
  (`महाराष्ट्र`, `ગુજરાત`, `ಕರ್ನಾಟಕ`, `পশ্চিমবঙ্গ`, `राजस्थान`, `मध्य प्रदेश`). Needs a small
  state-name dictionary built from training data itself (not an external gazetteer).
- **US state format varies systematically by source** — S2 tends abbreviated (`TX`), S3 tends
  full name (`Texas`) in the sample. Treat as a source-level normalization rule, not noise to
  discard.
- **Landmark references** (`Near SBI ATM`, `Opp.Rta Office`) are real, per the problem statement.
  Down-weight, don't rely on them.
- **Digit runs are everywhere and are comparison gold:** house numbers, survey numbers, PIN/unit
  numbers (`H.No.16-11-23/37/A`, `Sy.No.1009`) survive reformatting and reordering far better than
  surrounding text — see numeric-signature feature in §5.

### Country field

Reliable and clean in every sample row — always `US`/`India`, no blanks, no corruption. Use as a
**soft blocking/feature signal**, never a hard filter: the problem statement explicitly warns
against this, and since France appears only in test, any component hard-coded to `{US, India}`
will silently break on it. Always keep a small cross-country candidate pool per S1 as a safety
net against mislabeled country.

---

## 3. Matching strategy — technique selection and rationale

| Technique | Verdict | Why |
|---|---|---|
| Exact/normalized string match | Use — cheap tier-1 key | Instant easy wins, near-zero cost |
| Character n-gram (3-gram) TF-IDF / Jaccard | Use — core of candidate generation | Directly counters observed corruption (`Medf0rd`, `Cárolina`) and reordering; language-agnostic, so it also partially covers transliteration overlap |
| Token Jaccard / token-set ratio | Use as feature | Handles duplicated/reordered tokens (`Crestline Crestline Clean`) |
| Soft-TF-IDF (Monge-Elkan, IDF-weighted token match) | Use as feature | Suppresses false positives from generic tokens ("Private," "Limited," "Inc," "Ventures," "Group," "Solutions") that dominate naive similarity without a hand-built stopword list — IDF weight is learned from the corpus itself |
| Levenshtein / Jaro-Winkler | Use as feature, post-blocking | Good for short-string typo distance; too slow as sole retrieval mechanism at scale |
| Numeric-signature (digit-run set) overlap | Use — high-precision feature | Near-deterministic signal for address matches; survives reformatting/reordering better than any text similarity |
| Phonetic (Soundex/Metaphone) | Skip | English-centric, actively harmful on multilingual/transliterated names here |
| Script transliteration (romanization) | Use — preprocessing, not just retrieval | Converts non-Latin names/addresses to Latin so the *same* lexical feature battery applies, instead of leaving cross-script pairs dependent on embeddings alone |
| Multilingual embeddings (MIT/Apache, ≤8B) | Use — targeted retrieval + feature channel | Complements transliteration; catches semantic-but-not-lexical matches, still useful after romanization |
| ANN retrieval (FAISS) | Use, embedding channel only | Needed for speed at scale; skip if corpus small enough for brute-force cosine |
| Gradient-boosted trees (LightGBM) | Use — core classifier | Best fit for tabular similarity features at this data size: fast, tunable, explainable, license-trivial, no GPU dependency |
| Cross-encoder / LLM as primary matcher | Do not use as core decision-maker | Slower, hard to calibrate to F0.5 thresholds per-entity, ≤8B ceiling limits quality edge over a well-featured GBM anyway |
| LLM (Qwen-class, ≤8B) as reranker on ambiguous band only | Worth it as stretch goal, after core pipeline works | Apply only to near-threshold pairs the GBM is genuinely uncertain about |

**Architecture verdict:** hybrid retrieval + engineered features + calibrated LightGBM, wrapped
in a two-stage cascade (§4) and a metric-aligned decision layer (§8), is the right primary
architecture — it matches the observed noise profile, respects the license/param constraint
trivially, and gives per-entity probability control essential for F0.5 optimization.

---

## 4. Candidate generation — the recall ceiling

If the true match is absent from the candidate set, no downstream model can recover it. This
layer determines your ceiling; get it right before touching the classifier.

```
S1 record
 ├─► Blocking A: normalized-name exact/near-exact match
 ├─► Blocking B: character 3-gram TF-IDF cosine top-K (name)
 ├─► Blocking C: character 3-gram TF-IDF cosine top-K (address)
 ├─► Blocking D: token overlap (name ∪ address tokens)
 ├─► Blocking E: multilingual embedding ANN top-K (name + romanized name)
 ├─► Blocking F: country-soft-prioritized re-ranking of A–E (never a hard filter)
 ↓
 Union → dedupe by (S1, candidate_id), retain provenance (which strategy + rank)
 ↓
 Intra-source near-duplicate clustering: run the same blocking/similarity machinery
 *within* S2 alone and *within* S3 alone, assigning each record a cluster id.
 Given the GT pattern (multiple S2 and S3 ids per S1), these are very likely repeat
 listings of the same business within one source. Clustering lets accept/reject
 decisions propagate consistently across near-duplicate siblings instead of a
 model picking one and missing its twin, and enables cluster-level aggregate
 features (best name in cluster, address union) that are more robust than any
 single noisy record.
 ↓
 Recall@K measurement against GT (train split only), per strategy and combined,
 broken out by source (S2/S3) and country. Gate: do not proceed to feature
 engineering until combined recall clears ~97% at the chosen K (K≈50–100/S1).
 ↓
 Stage-1 cheap prune: compute only cheap features (exact/normalized match flag,
 char n-gram cosine, numeric-signature Jaccard) on the full unioned pool; keep
 the top ~10–15 candidates per S1. Re-measure Recall@K after pruning to confirm
 negligible recall loss. This is what makes embeddings, soft-TF-IDF, and listwise
 features (all more expensive) affordable within the time/compute budget — they
 only run on survivors, not the full ~50-candidate pool.
 ↓
 This survivor pool IS candidate_pairs.tsv — the final pre-model candidate set.
```

Run every blocking strategy independently and union — given corruption *and* reordering *and*
cross-script *and* website-style names all present simultaneously, no single retrieval method
covers all failure modes; each strategy recovers a different one.

---

## 5. Feature engineering

**Name features:** exact/normalized equality, Levenshtein ratio, Jaro-Winkler, token Jaccard,
token-set ratio (handles reorder/duplication), char-3gram cosine, length diff, token-count diff,
digit-token match (catches `0`/`o` corruption), URL-stripped-name similarity, embedding cosine
(multilingual channel, on both raw and romanized name), soft-TF-IDF (IDF-weighted Monge-Elkan
token similarity — suppresses generic-token false positives without a hand-built stopword list),
longest-common-substring ratio.

**Address features:** same battery on the normalized full-address string, plus:
- **Numeric-signature Jaccard** — digit-run sets extracted from raw address, compared as
  unordered sets. Expect this among the top features by importance: highly robust to
  reformatting/reordering, rare false collisions.
- Component-level where extractable: city similarity, state similarity (post native-script
  normalization), postal-code exact-match flag (missing-aware), house/unit-number token match.
- "Address comparable" indicator flags on both sides — never let a missing address silently
  zero out the score.

**Cross-field features:** country agreement (soft feature, never a filter), name-sim × address-sim
interaction (product term — useful for GBM splits), count of independent retrieval strategies
that surfaced this pair (cross-strategy agreement is cheap and strong), rank position within each
retrieval strategy.

**Listwise / group-relative features (computed after stage-1 pruning, on survivors):** for each
candidate, its name-sim and address-sim **z-score, percentile rank, and gap-to-next-best relative
to the other candidates for the same S1**. Motivation: absolute similarity is inflated for many
candidates when a business name is generic, but the true match usually still stands out relative
to its siblings — this is a dimension plain pairwise features can't express.

Start with ~25–35 features, check LightGBM feature importance, prune/expand from there (§14
ablation plan).

---

## 6. Hard-negative mining

Construct explicitly, not just randomly:
- Same normalized-name-blocking-bucket pairs absent from GT (near-name-match, wrong entity) —
  the pairs most likely to fool a name-only signal.
- Same city/state, different name-root pairs.
- Embedding-retrieved-but-GT-absent pairs (semantically close, lexically different) — teaches
  the model embedding similarity alone isn't sufficient.
- Other surviving candidates for the same S1 that aren't the true match — free, zero extra
  retrieval cost, and directly relevant to the listwise features above.

**Leakage guard:** mine hard negatives only from the train-split candidate pool; never use
validation-split GT to select negatives; never let a hard negative for one S1 leak information
about a different S1 sharing the same batch.

---

## 7. Multi-match output logic

GT shows S1 routinely maps to 2–7 records spanning both sources — **do not argmax**. Pipeline:

1. LightGBM outputs a raw match probability per (S1, candidate) pair.
2. Isotonic (or Platt) calibration on a held-out fold, fit **separately for S2 vs. S3** if their
   calibration curves differ meaningfully (test empirically, don't assume).
3. Feed calibrated probabilities into the decision layer in §8.

---

## 8. NO_MATCH handling and the decision layer

Treat NO_MATCH as a first-class problem, not a threshold afterthought — singletons are worth a
full 1.0 in the macro-average, and any spurious accept on a true singleton costs the whole point.

**Singleton gate (upstream, separate model):** train a small classifier (or a second LightGBM
head) on S1-level aggregate features — top candidate probability, candidate count, score-
distribution spread/entropy, Recall@K-implied candidate quality — to directly predict
P(this S1 is a true singleton). If confident, short-circuit to an empty prediction rather than
trusting per-pair probability math on a pool that's structurally low-signal for this S1.

**Expected-F0.5-maximizing accept set (replaces flat thresholding):** for an S1's candidates with
calibrated probabilities sorted descending `p_1 ≥ p_2 ≥ ... ≥ p_n`:

- For each k in {0, ..., n}: expected precision = mean(p_1..p_k); expected recall estimated
  against expected total true matches (e.g., sum of all p_i, or a small regressor on match-count
  trained separately); compute expected F0.5 via the official formula directly.
- Accept k* = argmax over k of expected F0.5.
- This subsumes both thresholding and singleton behavior in one step derived directly from the
  scored metric — a fixed global or per-source threshold is a special case of this and strictly
  worse whenever the probability spread varies meaningfully by S1, which it will given variable
  candidate-pool difficulty.
- Validate: confirm k* tracks the empirically-optimal k (computed directly against real GT on
  the validation fold) reasonably closely before trusting it in production. If it doesn't track
  well, fall back to a simpler per-source tuned threshold for that source only.
- If S2 and S3 show materially different noise/duplication profiles in error analysis, allow
  source-specific probability inputs into the same selector rather than forcing one calibration.

---

## 9. S2 ↔ S3 relationship

Worth exploiting lightly, not architecturally: if two accepted-candidate-quality records — one
S2, one S3 — for the same S1 are also highly similar *to each other*, that's corroborating
evidence both are genuine duplicates of the same entity. Add "candidate agrees with another
strong candidate for the same S1" as a feature/rule, not a full joint clustering model.

Full S1-S2-S3 joint graph clustering is over-engineering for the time window — the intra-source
clustering in §4 already captures most of the practical benefit (consistent decisions across
near-duplicate S2/S3 siblings) at a fraction of the complexity. Only escalate to full joint
clustering if error analysis in Phase 9 specifically shows the simpler approach missing
recoverable pairs.

---

## 10. Leakage-safe validation

- Split by **S1 entity**, never by row — an S1's full candidate set (true matches + hard
  negatives) must land entirely in one fold.
- Unsupervised fitting (TF-IDF vocab, embedding index, IDF weights for soft-TF-IDF, mined
  abbreviation dictionary) may use the full corpus since it doesn't touch GT labels — but
  threshold/selector tuning, calibration, hard-negative selection, and singleton-gate training
  must strictly respect the GT-based train/val split.
- Hold out a final internal test slice, untouched until the very end, to catch
  threshold/selector overfitting from repeated tuning iterations.
- **Held-out-country generalization check:** since France appears only in test with zero
  training exposure, run a validation variant that hides one training country entirely (e.g.,
  train without India, validate India as if unseen; repeat hiding US) to catch any feature or
  normalization step that's secretly country-specific and would silently fail on France. Cheap
  insurance — run once, late in the schedule.

---

## 11. Metric optimization

Official metric: **macro-averaged F0.5 per S1 entity, singleton-inclusive.** Precision-dominant
(precision weighted 2× recall). Optimize thresholds/selectors directly against this metric on
validation — not against AUC, logloss, or accuracy, which don't reflect the asymmetric penalty.

Worked example from the problem statement: predicting 3 matches where 2 are correct against a
true set of 2 gives Precision = 2/3, Recall = 1.0, F0.5 = 0.714 — illustrates how a single
over-inclusive prediction meaningfully costs an otherwise-perfect entity, reinforcing why the
accept-set logic in §8 is worth building properly rather than defaulting to a loose threshold.

---

## 12. Architecture comparison

| | A: Fuzzy/lexical only | B: Hybrid retrieval + engineered features + LightGBM (cascade, as specified above) | C: Embedding-heavy | D: LLM/Qwen-heavy |
|---|---|---|---|---|
| Handles observed corruption/reorder | Partial | Yes | Partial (misses typo-level) | Yes, slow |
| Cross-script India names | No | Yes — embedding channel **and** transliteration-based lexical channel | Yes | Yes |
| Generic-name false positives | High risk | Mitigated via soft-TF-IDF + listwise features | High risk | Low risk, costly |
| Singleton precision | Threshold-dependent, fragile | Dedicated gate | Threshold-dependent | Costly to calibrate |
| F0.5-metric alignment | Poor | Direct — expected-F0.5 selector | Poor | Hard to calibrate per-entity |
| Robustness to unseen country (France) | Depends on features used | Testable via holdout check, no hard filters anywhere | Depends on features used | Depends on features used |
| Compute cost at scale | Low | Medium, controlled by cascade | High | Very high |
| License/param fit (≤8B, MIT/Apache) | Trivial | Trivial | Trivial | Tight |
| Implementation risk in 72h | Low | Medium — more moving parts, but each piece is independently simple and testable | Medium | High |

**B wins on every axis that matters here.** Do not choose a sophisticated model merely because it
is sophisticated — this dataset's noise (corruption, reorder, cross-script, generic names,
within-source duplication) is exactly what a well-featured GBM with the right preprocessing
handles, and it stays controllable, explainable, and cheap.

---

## 13. LLM / Qwen role

Not the core matcher. Best use, time permitting: **reranker on the ambiguous probability band**
(e.g., calibrated score 0.35–0.65) where the GBM is genuinely uncertain — a small ≤8B MIT/Apache
model (Qwen2.5-class), given S1 name/address and candidate name/address, asked a constrained
yes/no with brief reasoning, can resolve cases lexical+embedding features can't fully separate
(legitimate abbreviation chains, ambiguous DBA names). Also useful, offline and unscored, as an
**error-analysis assistant** to summarize failure clusters. Do not let it touch high-confidence
accepts/rejects — that's where the calibrated GBM + expected-F0.5 selector is already reliable
and far faster.

---

## 14. Error analysis and ablation

**Error analysis — track and tabulate:**
- False positives and false negatives, split by source (S2/S3), country (incl. France once
  visible), and calibrated-score band.
- False negatives split into "never retrieved" (candidate-generation failure) vs. "retrieved but
  scored too low" (feature/model failure) — these need different fixes.
- Singleton accuracy specifically (given its outsized scoring weight).
- Per-script name-match failure rate (isolates transliteration/embedding channel effectiveness).
- Intra-source-cluster consistency (are near-duplicate siblings getting consistent decisions?).

**Ablation order** (cheapest-informative first — stop the moment a step stops moving validation
F0.5):
baseline exact/normalized match → + n-gram blocking → + fuzzy/edit-distance features →
+ address features incl. numeric signature → + transliteration → + embeddings →
+ soft-TF-IDF → + listwise features → + hard negatives → + calibration →
+ singleton gate → + expected-F0.5 selector (vs. flat threshold, compare directly) →
+ intra-source clustering.

---

## 15. Compute and engineering constraints

- Cache normalized strings, n-gram vectors, embeddings, and romanized text to disk keyed by
  entity_id, to avoid recompute across pipeline runs.
- Batch embedding computation; use a CPU-friendly small multilingual encoder if no GPU, or a
  quantized model.
- FAISS (IVF or HNSW) only if brute-force cosine is too slow at the actual corpus size — check
  row counts first (§0); don't add ANN infrastructure the data doesn't need.
- Deterministic seeds and reproducible pipeline stages throughout — required for the methodology
  write-up and package audit.
- Keep exploratory/ablation code in `experiments/`, separate from the production `src/` pipeline
  that generates the submitted outputs.

---

## 16. Directory structure

```
business_entity_resolution/
├── data/
│   ├── raw/                      # dataset/train, dataset/test
│   └── processed/                # normalized + cached intermediate artifacts
├── src/
│   ├── preprocessing/
│   │   ├── normalize_name.py
│   │   ├── normalize_address.py
│   │   ├── transliterate.py
│   │   ├── numeric_signature.py
│   │   └── abbreviation_miner.py
│   ├── candidate_generation/
│   │   ├── blocking_exact.py
│   │   ├── blocking_ngram.py
│   │   ├── blocking_token.py
│   │   ├── blocking_embedding.py
│   │   ├── intra_source_clustering.py
│   │   ├── union_and_dedupe.py
│   │   └── recall_at_k.py
│   ├── features/
│   │   ├── name_features.py
│   │   ├── address_features.py
│   │   ├── cross_features.py
│   │   ├── soft_tfidf.py
│   │   └── listwise_features.py
│   ├── models/
│   │   ├── stage1_prune.py
│   │   ├── pairwise_gbm.py
│   │   ├── singleton_gate.py
│   │   ├── calibration.py
│   │   └── llm_reranker.py         # stretch goal
│   ├── decision/
│   │   ├── expected_f05_selector.py
│   │   └── multi_accept.py
│   ├── evaluation/
│   │   ├── f05_scorer.py
│   │   ├── country_holdout_check.py
│   │   └── error_analysis.py
│   ├── inference/
│   │   └── run_pipeline.py
│   └── utils/
│       ├── caching.py
│       └── io.py
├── experiments/                  # ablations/notebooks, not the production path
├── outputs/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── configs/
│   └── pipeline.yaml
├── requirements.txt
└── README.md
```

---

## 17. Pipeline (end to end)

```
Raw TSVs
 → Validation (schema, tab-sep, id-prefix sanity)
 → Normalization
     - lowercase, junk-symbol/prefix strip (--, ***, #, @)
     - legal-suffix canonicalization (seed list + mined abbreviation dictionary)
     - URL-style name handling (strip TLD, split on . and -)
     - NULL/"null"/blank literal → true missing
     - native-script state name → canonical token (training-data-derived dictionary)
     - transliteration: script → Latin romanization for name AND address
     - numeric-signature extraction (digit runs from address)
 → Indexing
     - char n-gram TF-IDF index over normalized names/addresses
     - multilingual embedding index (ANN if corpus large; brute-force if small)
     - intra-source near-duplicate clustering on S2, on S3, independently
 → Candidate Generation (union of all blocking strategies, per S1, with provenance)
 → Recall@K measurement (gate: must clear ceiling before proceeding)
 → Stage-1 cheap prune (cheap features only; ~50 → ~10-15 survivors/S1; re-check recall)
 → Feature Engineering (full battery, survivors only)
     - pairwise name/address/cross features incl. numeric-signature Jaccard
     - soft-TF-IDF (generic-token down-weighted similarity)
     - listwise/group-relative features (z-score vs. this S1's other candidates)
 → Stage-2 Pairwise LightGBM → raw match probability
 → Calibration (isotonic; separately for S2 vs S3 if profiles differ)
 → Singleton gate (P(this S1 has zero true matches), trained on S1-level aggregates)
 → Expected-F0.5-maximizing accept-set selection (per S1)
 → (optional, time permitting) LLM reranker on residual ambiguous-band pairs
 → Output formatting: candidate_pairs.tsv (stage-1 survivor pool) + matching_results.tsv (final)
 → validate_submission.py
 → Country-holdout generalization check (proxy for unseen France)
```

---

## 18. 72-hour execution plan

| Phase | Hours | Priority | Notes |
|---|---|---|---|
| 0. Pre-flight data audit (§1 ambiguities, real files) | 0–3 | MUST | Do not skip |
| 1. Normalization incl. transliteration, numeric signature, mined abbreviations | 3–11 | MUST | |
| 2. Candidate generation, all blocking strategies + intra-source clustering | 11–23 | MUST | Measure Recall@K before moving on |
| 3. Stage-1 cheap prune + post-prune Recall@K re-check | 23–27 | MUST | Confirm pruning doesn't cost recall |
| 4. Feature engineering (full battery incl. soft-TF-IDF, listwise) | 27–36 | MUST | |
| 5. Stage-2 GBM training + calibration | 36–44 | MUST | |
| 6. Singleton gate training | 44–47 | SHOULD | High value-to-effort ratio |
| 7. Expected-F0.5 selector, validated against empirical optimum | 47–52 | MUST | Core of the decision layer |
| 8. Hard-negative mining pass + retrain | 52–57 | SHOULD | |
| 9. Error analysis + targeted fixes | 57–63 | SHOULD | |
| 10. Country-holdout generalization check | 63–65 | SHOULD | Cheap insurance vs. France |
| 11. Validation submission + format checks | 65–68 | MUST | |
| 12. LLM reranker on ambiguous band | 68–70 | STRETCH | Only if time remains |
| 13. Packaging, README, methodology doc | 70–72 | MUST | |

---

## 19. Antigravity TODO list (copy-paste ready)

```
TODO 1: Parse train_source1/2/3.tsv and train_ground_truth.tsv with sep="\t"; validate schema
        and id-prefix consistency. Report row counts per source and per country.

TODO 2: Explode train_ground_truth.tsv matched_entity_ids into (source1_id, matched_id) pairs.
        Compute: singleton rate, mean/median/max matches per S1, matched-id reuse across
        multiple S1 rows (flag if >0), source split of matches (S2-only / S3-only / both).
        Spot-check 20 singleton S1 rows manually against S2/S3 for unlabeled near-matches to
        assess GT exhaustiveness.

TODO 3: Implement normalize_name.py: lowercase, strip junk-prefix symbols (--, ***, #, @),
        canonicalize legal suffixes using a seed list, split/clean URL-style names
        (strip TLD, tokenize on . and -), strip literal "null"/"NULL" tokens.

TODO 4: Implement normalize_address.py: strip literal null tokens, canonicalize native-script
        state names to English tokens (dictionary built from train data only), canonicalize
        common abbreviations (Rd/Road, St/Street), preserve component text as an unordered
        bag of tokens (no positional parsing — component order is inconsistent even within
        one source).

TODO 5: Implement numeric_signature.py: extract all digit runs (regex \d+) from raw address
        text per record; store as a set for later Jaccard-overlap feature computation.

TODO 6: Implement transliterate.py: apply algorithmic script-to-Latin romanization to
        business_name and business_address for non-Latin-script records; store romanized
        version alongside original for downstream lexical feature computation (do not discard
        the original — some names mix scripts).

TODO 7: Implement abbreviation_miner.py: cluster near-identical names (edit distance below a
        small threshold) that differ in exactly one token across the training corpus;
        aggregate frequent differing-token pairs into a mined abbreviation dictionary; merge
        into the suffix/abbreviation canonicalization used in TODO 3.

TODO 8: Implement blocking_exact.py, blocking_ngram.py (char 3-gram TF-IDF top-K),
        blocking_token.py (token-overlap top-K) over normalized name+address, separately for
        name and address.

TODO 9: Implement blocking_embedding.py: multilingual sentence-embedding index (MIT/Apache,
        ≤8B-compatible small encoder) over normalized name and romanized name; brute-force
        cosine if corpus small, FAISS ANN if large.

TODO 10: Implement intra_source_clustering.py: run the same blocking/similarity machinery
         within S2 alone and within S3 alone to detect near-duplicate listings; assign a
         cluster id to each S2/S3 record.

TODO 11: Implement union_and_dedupe.py: union all candidate sources into one pool per S1,
         dedupe by (s1_id, candidate_id), retain provenance (which strategy retrieved it, its
         rank in that strategy) as future features.

TODO 12: Implement recall_at_k.py: on the train split, measure Recall@K per strategy and
         combined against exploded ground truth, broken out by source (S2/S3) and country.
         Gate: do not proceed to feature engineering until combined recall clears ~97% at
         the chosen K.

TODO 13: Implement stage1_prune.py: compute cheap features (exact/normalized match flag,
         char-ngram cosine, numeric-signature Jaccard) on the full candidate pool; keep top
         ~10-15 candidates per S1 by a simple weighted cheap-score. Re-run recall_at_k after
         pruning to confirm minimal recall loss.

TODO 14: Implement name_features.py, address_features.py, cross_features.py: full pairwise
         similarity battery (edit distance, Jaro-Winkler, token Jaccard, char n-gram cosine,
         length/token-count diffs, embedding cosine, numeric-signature Jaccard, missing-field
         indicators) on stage-1 survivors only.

TODO 15: Implement soft_tfidf.py: fit token IDF weights over the full training corpus (S1+S2+S3
         combined, unsupervised); compute Monge-Elkan-style soft-TF-IDF similarity for name
         pairs.

TODO 16: Implement listwise_features.py: for each surviving candidate, compute its
         name-sim/address-sim z-score, percentile rank, and gap-to-next-best relative to other
         candidates for the same S1.

TODO 17: Implement pairwise_gbm.py: train LightGBM binary classifier on labeled pairs (true
         match / hard negative) using S1-grouped train/validation split. Log feature
         importances.

TODO 18: Implement hard-negative mining: same-block non-matches, same-city different-name
         pairs, embedding-retrieved-but-unmatched pairs, sibling candidates for the same S1
         that aren't in GT. Mine only from the train-split candidate pool.

TODO 19: Implement calibration.py: fit isotonic regression on validation-fold predictions; fit
         separately for S2-sourced vs. S3-sourced candidates if their calibration curves
         differ meaningfully.

TODO 20: Implement singleton_gate.py: train a classifier on S1-level aggregate features (top
         candidate probability, candidate count, score spread) to predict P(true singleton);
         validate its precision specifically on singleton-accuracy contribution to F0.5.

TODO 21: Implement expected_f05_selector.py: given calibrated probabilities sorted descending
         for an S1's candidates, compute expected F0.5 for each accept-count k using the F0.5
         formula directly; select k* = argmax. Validate k* tracks the empirically F0.5-optimal
         k on a held-out fold; fall back to a tuned per-source threshold if it doesn't track
         well.

TODO 22: Implement multi_accept.py: combine singleton_gate output and expected_f05_selector
         output into the final per-S1 accepted candidate list; enforce output constraints
         (S2/S3 ids only, no duplicates, ids must exist in test set).

TODO 23: Implement f05_scorer.py: compute per-S1 F0.5 and macro-average, matching the official
         formula exactly, for use in all validation loops.

TODO 24: Implement error_analysis.py: generate false-positive/false-negative tables split by
         source, country, script, and score band; singleton-accuracy breakdown;
         "retrieved but scored low" vs. "never retrieved" false-negative split; intra-source
         cluster consistency check.

TODO 25: Implement country_holdout_check.py: retrain/validate with one training country hidden
         at a time; report F0.5 degradation as a proxy for France-generalization risk.

TODO 26: Implement run_pipeline.py: end-to-end inference script producing
         output/candidate_pairs.tsv (stage-1-survivor pool, pre-final-model) and
         output/matching_results.tsv (final accepted matches), for both validation and test
         data.

TODO 27: Run utils/validate_submission.py against generated outputs before any leaderboard
         upload; fix any reported issues.

TODO 28: Write README.md with exact reproduction steps and requirements.txt with pinned
         versions; confirm final model choice is MIT/Apache-licensed and ≤8B parameters.

TODO 29: Fill in Documentation_template.md describing methodology, blocking strategy, model
         architecture, feature engineering, and every enhancement in this document.
```

---

## 20. Final recommendation

**A. Recommended architecture:** Multi-strategy candidate generation (exact + n-gram + token +
multilingual embedding, unioned, country-soft, plus intra-source near-duplicate clustering) →
stage-1 cheap prune → full engineered feature battery (incl. numeric signature, soft-TF-IDF,
listwise features) on survivors → calibrated LightGBM → singleton gate → expected-F0.5-maximizing
per-S1 accept-set selection. Optional LLM reranker only on the score-ambiguous band, added last.

**B. Why this architecture:** It is the only option in §12 that simultaneously handles the
data's actual noise profile (corruption, reorder, cross-script, generic names, within-source
duplication), respects the license/parameter ceiling trivially, and optimizes directly against
the scored metric rather than a proxy for it — while staying implementable and debuggable inside
72 hours.

**C. Components deliberately not built:** full joint S1-S2-S3 graph clustering (intra-source
clustering captures most of the benefit far more cheaply); phonetic matching (harmful given the
multilingual/transliterated profile); LLM as primary matcher; hard country filtering anywhere;
positional/rule-based address parsing.

**D. Biggest risks, in practical order:** (1) the training-GT-to-sample mismatch means blocking
recall on the real data is currently unmeasured — verify immediately in Phase 0; (2) an
uncalibrated or poorly-validated decision layer under F0.5's precision weighting can tank score
even with a good underlying model — this is why §8's selector gets validated against empirical
optimum, not trusted blindly; (3) unhandled native-script state names and URL-style names
silently degrading a meaningful chunk of records if normalization is incomplete; (4) unseen-
country (France) failure from any accidentally country-specific feature or rule.

**E. Biggest opportunities:** the URL-as-name and duplicated-token patterns are mechanical and
cheaply detectable — targeted normalization rules here likely buy more F0.5 than extra model
tuning; the numeric-signature feature is a near-free, high-precision address signal; the
expected-F0.5 selector directly targets what's actually scored instead of a proxy threshold.

**F. Priority order for the 72 hours:** data audit → normalization (incl. transliteration,
numeric signature) → candidate generation + Recall@K gate → stage-1 prune → feature engineering
→ GBM + calibration → singleton gate → expected-F0.5 selector → hard negatives → error analysis
→ country-holdout check → validated submission → (stretch) LLM reranker → packaging.

**G. Final TODO list:** §19, above — copy-paste ready for Antigravity.
