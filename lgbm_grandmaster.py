import os
import sys
import time
import gc
import re
from pathlib import Path
import polars as pl
import numpy as np
import lightgbm as lgb
from rapidfuzz import distance, fuzz

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.config import TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GT_PATH, TEST_S1_PATH, TEST_S2_PATH, TEST_S3_PATH

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORP_PREFIXES = r"^(dr|shree|sri|om|m\/s|ms|hotel|the|saint|st|maison|atelier|ste|boulangerie|patisserie|pharmacie|cabinet|garage|boutique|agence|auto|restaurant|bar)\s+"
CORP_SUFFIXES = r"\b(inc|incorporated|corp|corporation|llc|ltd|limited|co|company|gmbh|sarl|sas|sasu|sa|sci|eurl|snc|gie|cie|societe|pvt|pvt ltd|private limited|private ltd|private|group|holdings|services|solutions|enterprises|technologies|associates|partners|international|intl|d\.b\.a\.|dba|fils|et fils|\& fils|freres|et cie)\b"

SOUNDEX_MAP = {
    'b': '1', 'f': '1', 'p': '1', 'v': '1',
    'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
    'd': '3', 't': '3',
    'l': '4',
    'm': '5', 'n': '5',
    'r': '6'
}

def compute_soundex(word: str) -> str:
    if not word or not isinstance(word, str) or len(word) == 0:
        return ""
    w = word.lower()
    first_char = w[0]
    if not ('a' <= first_char <= 'z'):
        return ""
    code = [first_char]
    prev = SOUNDEX_MAP.get(first_char, '0')
    for ch in w[1:]:
        c = SOUNDEX_MAP.get(ch, '0')
        if c != '0' and c != prev:
            code.append(c)
        prev = c
        if len(code) == 4:
            break
    while len(code) < 4:
        code.append('0')
    return "".join(code[:4])

def compute_consonant_stem(word: str) -> str:
    if not word or not isinstance(word, str) or len(word) == 0:
        return ""
    w = word.lower()
    consonants = [ch for ch in w if 'a' <= ch <= 'z' and ch not in 'aeiou']
    return "".join(consonants[:3])

def compute_sorted_3gram(word: str) -> str:
    if not word or not isinstance(word, str) or len(word) < 3:
        return ""
    clean = "".join([c for c in word.lower() if 'a' <= c <= 'z'])
    if len(clean) < 3:
        return ""
    return "".join(sorted(clean[:4]))

def clean_text(expr: pl.Expr) -> pl.Expr:
    return (
        expr.fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[<>{}\[\]\(\)\-_.,;:!?'\"/\\|*~@#$%^&=+`]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )

def leet_clean_text(expr: pl.Expr) -> pl.Expr:
    return (
        clean_text(expr)
        .str.replace_all(r"(\b[a-z]+)0([a-z]+\b)", "${1}o${2}")
        .str.replace_all(r"(\b[a-z]+)1([a-z]+\b)", "${1}l${2}")
        .str.replace_all(r"(\b[a-z]+)3([a-z]+\b)", "${1}e${2}")
        .str.replace_all(r"(\b[a-z]+)5([a-z]+\b)", "${1}s${2}")
        .str.replace_all(r"\bph", "f")
    )

def standardize_address(expr: pl.Expr) -> pl.Expr:
    return (
        clean_text(expr)
        .str.replace_all(r"\br\b|\brue\b", "rue")
        .str.replace_all(r"\bav\b|\bave\b|\bavenue\b", "ave")
        .str.replace_all(r"\bbd\b|\bblvd\b|\bboulevard\b", "blvd")
        .str.replace_all(r"\ball\b|\ballee\b|\ballée\b", "allee")
        .str.replace_all(r"\bpl\b|\bplace\b", "place")
        .str.replace_all(r"\bno\b|\bn°\b", "")
        .str.replace_all(r"\bstreet\b|\bsaint\b", "st")
        .str.replace_all(r"\broad\b", "rd")
        .str.replace_all(r"\bdrive\b", "dr")
        .str.replace_all(r"\blane\b", "ln")
        .str.replace_all(r"\bsuite\b|\bste\b|\bshop no\b|\bshop\b|\bfl\b|\bfloor\b", "ste")
        .str.replace_all(r"\bhighway\b|\bhwy\b", "hwy")
        .str.replace_all(r"\bapartment\b|\bapt\b", "apt")
        .str.replace_all(r"\bsector\b|\bsec\b", "sec")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )

def clean_address_core(expr: pl.Expr) -> pl.Expr:
    return (
        standardize_address(expr)
        .str.replace_all(r"\b(st|rd|dr|ln|ave|blvd|ste|apt|hwy|unit|floor|fl|plot|sec|sector|near|opp|behind|nagar|colony|road|street|marg|bordeaux|paris|lille|lyon|mumbai|delhi|bangalore|hyderabad|chennai)\b", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )

def standardize_country(expr: pl.Expr) -> pl.Expr:
    return (
        clean_text(expr)
        .str.replace_all(r"\b(united states|usa|u s a|america)\b", "us")
        .str.replace_all(r"\b(united kingdom|great britain|uk|england)\b", "gb")
        .str.replace_all(r"\b(india|ind)\b", "in")
        .str.replace_all(r"\b(france|republique francaise|republique)\b", "fr")
        .str.strip_chars()
    )

def strip_corp_affixes(expr: pl.Expr) -> pl.Expr:
    return (
        leet_clean_text(expr)
        .str.replace_all(CORP_PREFIXES, "")
        .str.replace_all(CORP_SUFFIXES, " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )

def get_sorted_root(expr: pl.Expr) -> pl.Expr:
    return (
        strip_corp_affixes(expr)
        .str.split(" ")
        .list.sort()
        .list.join(" ")
        .fill_null("")
    )

def extract_acronym(expr: pl.Expr) -> pl.Expr:
    return (
        strip_corp_affixes(expr)
        .str.split(" ")
        .list.eval(pl.element().str.slice(0, 1))
        .list.join("")
        .fill_null("")
    )

def extract_street_digits(expr: pl.Expr) -> pl.Expr:
    return expr.fill_null("").str.extract(r"(\d+)", 1).fill_null("")

def extract_postal_code(expr: pl.Expr) -> pl.Expr:
    return expr.fill_null("").str.extract(r"\b(\d{5,6})\b", 1).fill_null("")

def extract_unit_code(expr: pl.Expr) -> pl.Expr:
    return (
        clean_text(expr)
        .str.extract(r"\b(?:ste|suite|unit|flat|office|fl|floor|plot|shop|bldg)\s*(\d+)\b", 1)
        .fill_null("")
    )

def extract_features_v27_chunk(n1_arr, n2_arr, r1_arr, r2_arr, ac1_arr, ac2_arr, a1_arr, a2_arr, acore1_arr, acore2_arr, d1_arr, d2_arr, u1_arr, u2_arr, p1_arr, p2_arr, c1_arr, c2_arr, sx1_arr, sx2_arr, cs1_arr, cs2_arr):
    n = len(n1_arr)
    X = np.zeros((n, 40), dtype=np.float32)

    for i in range(n):
        n1, n2 = n1_arr[i], n2_arr[i]
        r1, r2 = r1_arr[i], r2_arr[i]
        ac1, ac2 = ac1_arr[i], ac2_arr[i]
        a1, a2 = a1_arr[i], a2_arr[i]
        acore1, acore2 = acore1_arr[i], acore2_arr[i]
        d1, d2 = d1_arr[i], d2_arr[i]
        u1, u2 = u1_arr[i], u2_arr[i]
        p1, p2 = p1_arr[i], p2_arr[i]
        c1, c2 = c1_arr[i], c2_arr[i]
        sx1, sx2 = sx1_arr[i], sx2_arr[i]
        cs1, cs2 = cs1_arr[i], cs2_arr[i]

        exact_name = 1.0 if (n1 and n2 and n1 == n2) else 0.0
        exact_root = 1.0 if (r1 and r2 and r1 == r2) else 0.0

        jw_root = distance.JaroWinkler.similarity(r1, r2) if (r1 and r2) else 0.0
        sort_root = fuzz.token_sort_ratio(r1, r2) / 100.0 if (r1 and r2) else 0.0
        set_root = fuzz.token_set_ratio(r1, r2) / 100.0 if (r1 and r2) else 0.0
        partial_root = fuzz.partial_ratio(r1, r2) / 100.0 if (r1 and r2) else 0.0
        lev_root = distance.Levenshtein.normalized_similarity(r1, r2) if (r1 and r2) else 0.0
        qratio_root = fuzz.QRatio(r1, r2) / 100.0 if (r1 and r2) else 0.0

        len_r1, len_r2 = len(r1), len(r2)
        len_diff_ratio = abs(len_r1 - len_r2) / (max(len_r1, len_r2, 1))

        acronym_match = 1.0 if ((ac1 and r2 and (ac1 == r2 or ac1 == ac2)) or (ac2 and r1 and (ac2 == r1))) else 0.0
        soundex_match = 1.0 if (sx1 and sx2 and sx1 == sx2) else 0.0
        consonant_match = 1.0 if (cs1 and cs2 and cs1 == cs2 and len(cs1) >= 2) else 0.0
        jw_soundex = distance.JaroWinkler.similarity(sx1, sx2) if (sx1 and sx2) else 0.0

        jw_addr = distance.JaroWinkler.similarity(a1, a2) if (a1 and a2) else 0.40
        sort_addr = fuzz.token_sort_ratio(a1, a2) / 100.0 if (a1 and a2) else 0.40
        set_addr = fuzz.token_set_ratio(a1, a2) / 100.0 if (a1 and a2) else 0.40
        lev_addr = distance.Levenshtein.normalized_similarity(a1, a2) if (a1 and a2) else 0.40
        core_addr_sort = fuzz.token_sort_ratio(acore1, acore2) / 100.0 if (acore1 and acore2) else 0.40
        core_addr_set = fuzz.token_set_ratio(acore1, acore2) / 100.0 if (acore1 and acore2) else 0.40
        core_addr_lev = distance.Levenshtein.normalized_similarity(acore1, acore2) if (acore1 and acore2) else 0.40

        same_digits = 1.0 if (d1 and d2 and d1 == d2) else 0.0
        diff_digits = 1.0 if (d1 and d2 and d1 != d2) else 0.0
        digit_num_diff = 0.0
        adjacent_door_penalty = 0.0
        if d1 and d2 and d1.isdigit() and d2.isdigit():
            val1, val2 = int(d1), int(d2)
            delta = abs(val1 - val2)
            digit_num_diff = min(delta, 50) / 50.0
            if 1 <= delta <= 4:
                adjacent_door_penalty = 1.0

        same_unit = 1.0 if (u1 and u2 and u1 == u2) else 0.0
        same_pin = 1.0 if (p1 and p2 and p1 == p2) else 0.0
        diff_pin = 1.0 if (p1 and p2 and p1 != p2) else 0.0
        same_country = 1.0 if (c1 and c2 and c1 == c2) else 0.0

        translit_flag = 1.0 if (sort_addr >= 0.70 and same_digits == 1.0) else 0.0
        composite = max(jw_root, sort_root, acronym_match) * sort_addr
        addr_match_score = (sort_addr * 0.7) + (same_digits * 0.3)
        pin_and_digits = 1.0 if (same_pin == 1.0 and same_digits == 1.0) else 0.0
        phonetic_addr_composite = max(soundex_match, consonant_match) * sort_addr
        strong_exact_pair = 1.0 if (exact_root == 1.0 and (same_digits == 1.0 or same_pin == 1.0)) else 0.0
        
        root_contains = 1.0 if (r1 and r2 and len(r1) >= 4 and len(r2) >= 4 and (r1 in r2 or r2 in r1)) else 0.0
        exact_full_addr = 1.0 if (a1 and a2 and a1 == a2) else 0.0
        exact_root_and_pin = 1.0 if (exact_root == 1.0 and same_pin == 1.0) else 0.0
        exact_addr_core = 1.0 if (acore1 and acore2 and acore1 == acore2 and len(acore1) >= 6) else 0.0
        
        tokens1 = set(r1.split()) if r1 else set()
        tokens2 = set(r2.split()) if r2 else set()
        shared_tokens = len(tokens1 & tokens2) / max(len(tokens1 | tokens2), 1)
        sort_set_diff = abs(sort_root - set_root)

        X[i, 0] = exact_name
        X[i, 1] = exact_root
        X[i, 2] = jw_root
        X[i, 3] = sort_root
        X[i, 4] = set_root
        X[i, 5] = partial_root
        X[i, 6] = lev_root
        X[i, 7] = qratio_root
        X[i, 8] = len_diff_ratio
        X[i, 9] = acronym_match
        X[i, 10] = soundex_match
        X[i, 11] = consonant_match
        X[i, 12] = jw_soundex
        X[i, 13] = jw_addr
        X[i, 14] = sort_addr
        X[i, 15] = set_addr
        X[i, 16] = lev_addr
        X[i, 17] = core_addr_sort
        X[i, 18] = core_addr_set
        X[i, 19] = core_addr_lev
        X[i, 20] = same_digits
        X[i, 21] = diff_digits
        X[i, 22] = digit_num_diff
        X[i, 23] = same_unit
        X[i, 24] = same_pin
        X[i, 25] = diff_pin
        X[i, 26] = same_country
        X[i, 27] = translit_flag
        X[i, 28] = composite
        X[i, 29] = addr_match_score
        X[i, 30] = pin_and_digits
        X[i, 31] = phonetic_addr_composite
        X[i, 32] = strong_exact_pair
        X[i, 33] = root_contains
        X[i, 34] = exact_full_addr
        X[i, 35] = exact_root_and_pin
        X[i, 36] = exact_addr_core
        X[i, 37] = shared_tokens
        X[i, 38] = adjacent_door_penalty
        X[i, 39] = sort_set_diff

    return X

def prepare_df(path, is_s1=False):
    raw = pl.read_csv(path, separator="\t", truncate_ragged_lines=True, ignore_errors=True)
    id_c = "entity_id" if "entity_id" in raw.columns else raw.columns[0]
    id_alias = "s1_id" if is_s1 else "cand_id"
    df = raw.select([
        pl.col(id_c).cast(pl.Utf8).alias(id_alias),
        clean_text(pl.col("business_name")).alias("name"),
        strip_corp_affixes(pl.col("business_name")).alias("name_root"),
        get_sorted_root(pl.col("business_name")).alias("sorted_root"),
        strip_corp_affixes(pl.col("business_name")).str.split(" ").list.get(0, null_on_oob=True).fill_null("").alias("first_word"),
        strip_corp_affixes(pl.col("business_name")).str.split(" ").list.get(1, null_on_oob=True).fill_null("").alias("second_word"),
        extract_acronym(pl.col("business_name")).alias("acronym"),
        clean_text(pl.col("business_address")).alias("addr"),
        standardize_address(pl.col("business_address")).alias("addr_std"),
        clean_address_core(pl.col("business_address")).alias("addr_core"),
        clean_text(pl.col("country")).alias("country"),
        standardize_country(pl.col("country")).alias("country_std"),
        extract_street_digits(pl.col("business_address")).alias("street_digits"),
        extract_unit_code(pl.col("business_address")).alias("unit_code"),
        extract_postal_code(pl.col("business_address")).alias("postal_code"),
        strip_corp_affixes(pl.col("business_name")).str.slice(0, 5).alias("root_prefix_5"),
        strip_corp_affixes(pl.col("business_name")).str.slice(0, 4).alias("root_prefix"),
        strip_corp_affixes(pl.col("business_name")).str.slice(0, 3).alias("root_prefix_3"),
        strip_corp_affixes(pl.col("business_name")).str.slice(0, 2).alias("root_prefix_2"),
        clean_address_core(pl.col("business_address")).str.slice(0, 4).alias("addr_core_prefix_4")
    ])
    
    first_words = df["first_word"].to_list()
    second_words = df["second_word"].to_list()
    roots = df["name_root"].to_list()
    
    soundex_w1 = [compute_soundex(w) for w in first_words]
    soundex_w2 = [compute_soundex(w) for w in second_words]
    consonant_stems = [compute_consonant_stem(w) for w in first_words]
    sorted_3grams = [compute_sorted_3gram(r) for r in roots]
    
    return df.with_columns([
        pl.Series("soundex_first", soundex_w1),
        pl.Series("soundex_second", soundex_w2),
        pl.Series("consonant_stem", consonant_stems),
        pl.Series("sorted_3gram", sorted_3grams)
    ])

def train_ensemble():
    print("=" * 90)
    print("🧠 SOTA 0.90+ GBDT GRANDMASTER PRODUCTION TRAINING (5-Seed Quintuple Ensemble)")
    print("=" * 90)
    t0 = time.time()

    s1_all = prepare_df(TRAIN_S1_PATH, is_s1=True)
    s2_all = prepare_df(TRAIN_S2_PATH)
    s3_all = prepare_df(TRAIN_S3_PATH)

    df_gt = pl.read_csv(TRAIN_GT_PATH, separator="\t", truncate_ragged_lines=True, ignore_errors=True)
    gt_map = {}
    for row in df_gt.iter_rows():
        s1 = str(row[0])
        val = str(row[1]) if row[1] is not None else ""
        if val.strip() in ("", "[]", "None"):
            gt_map[s1] = set()
        else:
            gt_map[s1] = {x.strip() for x in val.split(",") if x.strip()}

    sat_all = pl.concat([s2_all, s3_all]).unique(subset=["cand_id"])
    s1_train_ids = [sid for sid in s1_all["s1_id"].to_list() if sid in gt_map][:130_000]

    pos_pairs = []
    for s1_id in s1_train_ids:
        for c in gt_map.get(s1_id, []):
            pos_pairs.append((s1_id, c, 1.0))
            if len(pos_pairs) >= 120_000:
                break
        if len(pos_pairs) >= 120_000:
            break

    p_neg1 = s1_all.head(50_000).join(s2_all.head(130_000), on=["name_root", "country_std"], how="inner").head(80_000).select(["s1_id", "cand_id"])
    p_neg2 = s1_all.head(50_000).join(s2_all.head(130_000), on=["street_digits", "root_prefix_3", "country_std"], how="inner").head(60_000).select(["s1_id", "cand_id"])
    
    neg_pairs = []
    for row in pl.concat([p_neg1, p_neg2]).unique().iter_rows():
        s1, c = row
        if c not in gt_map.get(s1, set()):
            neg_pairs.append((s1, c, 0.0))

    all_pairs = pos_pairs + neg_pairs
    df_pairs = pl.DataFrame({
        "s1_id": [p[0] for p in all_pairs],
        "cand_id": [p[1] for p in all_pairs],
        "label": [p[2] for p in all_pairs]
    })

    joined = df_pairs.join(
        s1_all.select([
            pl.col("s1_id"),
            pl.col("name").alias("s1_name"),
            pl.col("name_root").alias("s1_root"),
            pl.col("acronym").alias("s1_acronym"),
            pl.col("addr_std").alias("s1_addr"),
            pl.col("addr_core").alias("s1_addr_core"),
            pl.col("street_digits").alias("s1_digits"),
            pl.col("unit_code").alias("s1_unit"),
            pl.col("postal_code").alias("s1_pin"),
            pl.col("country_std").alias("s1_country"),
            pl.col("soundex_first").alias("s1_sx"),
            pl.col("consonant_stem").alias("s1_cs")
        ]),
        on="s1_id", how="left"
    ).join(
        sat_all.select([
            pl.col("cand_id"),
            pl.col("name").alias("cand_name"),
            pl.col("name_root").alias("cand_root"),
            pl.col("acronym").alias("cand_acronym"),
            pl.col("addr_std").alias("cand_addr"),
            pl.col("addr_core").alias("cand_addr_core"),
            pl.col("street_digits").alias("cand_digits"),
            pl.col("unit_code").alias("cand_unit"),
            pl.col("postal_code").alias("cand_pin"),
            pl.col("country_std").alias("cand_country"),
            pl.col("soundex_first").alias("cand_sx"),
            pl.col("consonant_stem").alias("cand_cs")
        ]),
        on="cand_id", how="left"
    )

    X_train = extract_features_v27_chunk(
        joined["s1_name"].to_numpy(), joined["cand_name"].to_numpy(),
        joined["s1_root"].to_numpy(), joined["cand_root"].to_numpy(),
        joined["s1_acronym"].to_numpy(), joined["cand_acronym"].to_numpy(),
        joined["s1_addr"].to_numpy(), joined["cand_addr"].to_numpy(),
        joined["s1_addr_core"].to_numpy(), joined["cand_addr_core"].to_numpy(),
        joined["s1_digits"].to_numpy(), joined["cand_digits"].to_numpy(),
        joined["s1_unit"].to_numpy(), joined["cand_unit"].to_numpy(),
        joined["s1_pin"].to_numpy(), joined["cand_pin"].to_numpy(),
        joined["s1_country"].to_numpy(), joined["cand_country"].to_numpy(),
        joined["s1_sx"].to_numpy(), joined["cand_sx"].to_numpy(),
        joined["s1_cs"].to_numpy(), joined["cand_cs"].to_numpy()
    )
    y_train = joined["label"].to_numpy().astype(np.float32)

    train_data1 = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    train_data2 = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    train_data3 = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    train_data4 = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    train_data5 = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    
    params_lgb1 = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 85,
        "max_depth": 9,
        "scale_pos_weight": 1.30,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "reg_alpha": 0.05,
        "reg_lambda": 0.5,
        "seed": 42,
        "verbosity": -1,
        "num_threads": min(os.cpu_count() or 4, 8)
    }
    params_lgb2 = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.045,
        "num_leaves": 95,
        "max_depth": 10,
        "scale_pos_weight": 1.25,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "min_child_samples": 15,
        "reg_alpha": 0.10,
        "reg_lambda": 0.8,
        "seed": 1337,
        "verbosity": -1,
        "num_threads": min(os.cpu_count() or 4, 8)
    }
    params_lgb3 = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.055,
        "num_leaves": 75,
        "max_depth": 8,
        "scale_pos_weight": 1.35,
        "feature_fraction": 0.88,
        "bagging_fraction": 0.88,
        "bagging_freq": 1,
        "min_child_samples": 22,
        "reg_alpha": 0.08,
        "reg_lambda": 0.6,
        "seed": 2026,
        "verbosity": -1,
        "num_threads": min(os.cpu_count() or 4, 8)
    }
    params_lgb4 = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.04,
        "num_leaves": 105,
        "max_depth": 10,
        "scale_pos_weight": 1.20,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "min_child_samples": 18,
        "reg_alpha": 0.12,
        "reg_lambda": 0.9,
        "seed": 777,
        "verbosity": -1,
        "num_threads": min(os.cpu_count() or 4, 8)
    }
    params_lgb5 = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.035,
        "num_leaves": 120,
        "max_depth": 11,
        "scale_pos_weight": 1.25,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_child_samples": 16,
        "reg_alpha": 0.15,
        "reg_lambda": 1.0,
        "seed": 999,
        "verbosity": -1,
        "num_threads": min(os.cpu_count() or 4, 8)
    }

    bst1 = lgb.train(params_lgb1, train_data1, num_boost_round=200)
    bst2 = lgb.train(params_lgb2, train_data2, num_boost_round=220)
    bst3 = lgb.train(params_lgb3, train_data3, num_boost_round=180)
    bst4 = lgb.train(params_lgb4, train_data4, num_boost_round=240)
    bst5 = lgb.train(params_lgb5, train_data5, num_boost_round=260)
    print(f"  * 5-Seed Quintuple Ensemble Trained in {time.time() - t0:.2f}s")
    
    return bst1, bst2, bst3, bst4, bst5

def score_test_chunk(s1_chunk, sat_df, bst1, bst2, bst3, bst4, bst5):
    # 1. Exact Name
    p1 = s1_chunk.filter(pl.col("name").str.len_chars() >= 3).join(
        sat_df.filter(pl.col("name").str.len_chars() >= 3),
        on=["name", "country_std"], how="inner"
    ).group_by("s1_id").head(4).select(["s1_id", "cand_id"])

    # 2. Name Root
    p2 = s1_chunk.filter((pl.col("name_root").str.len_chars() >= 3) & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("name_root").str.len_chars() >= 3) & (pl.col("country_std") != "")),
        on=["name_root", "country_std"], how="inner"
    ).group_by("s1_id").head(4).select(["s1_id", "cand_id"])
    
    # 3. Sorted Root
    p3 = s1_chunk.filter((pl.col("sorted_root").str.len_chars() >= 4) & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("sorted_root").str.len_chars() >= 4) & (pl.col("country_std") != "")),
        on=["sorted_root", "country_std"], how="inner"
    ).group_by("s1_id").head(3).select(["s1_id", "cand_id"])

    # 4. Street Digits + Root Prefix 3
    p4 = s1_chunk.filter((pl.col("street_digits") != "") & (pl.col("root_prefix_3").str.len_chars() >= 3) & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("street_digits") != "") & (pl.col("root_prefix_3").str.len_chars() >= 3)),
        on=["street_digits", "root_prefix_3", "country_std"], how="inner"
    ).group_by("s1_id").head(4).select(["s1_id", "cand_id"])
    
    # 5. Unit Code + Root Prefix 3
    p5 = s1_chunk.filter((pl.col("unit_code") != "") & (pl.col("root_prefix_3").str.len_chars() >= 3) & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("unit_code") != "") & (pl.col("root_prefix_3").str.len_chars() >= 3)),
        on=["unit_code", "root_prefix_3", "country_std"], how="inner"
    ).group_by("s1_id").head(3).select(["s1_id", "cand_id"])

    # 6. Double-Anchor PIN Code + 3-Char Prefix
    p6 = s1_chunk.filter((pl.col("postal_code") != "") & (pl.col("root_prefix_3").str.len_chars() >= 3) & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("postal_code") != "") & (pl.col("root_prefix_3").str.len_chars() >= 3)),
        on=["postal_code", "root_prefix_3", "country_std"], how="inner"
    ).group_by("s1_id").head(4).select(["s1_id", "cand_id"])

    # 7. Indic Transliteration Double-Anchor: [postal_code, street_digits]
    p7 = s1_chunk.filter((pl.col("postal_code") != "") & (pl.col("street_digits") != "") & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("postal_code") != "") & (pl.col("street_digits") != "")),
        on=["postal_code", "street_digits", "country_std"], how="inner"
    ).group_by("s1_id").head(3).select(["s1_id", "cand_id"])

    # 8. Soundex First Word + PIN Code
    p8 = s1_chunk.filter((pl.col("soundex_first") != "") & (pl.col("postal_code") != "") & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("soundex_first") != "") & (pl.col("postal_code") != "")),
        on=["soundex_first", "postal_code", "country_std"], how="inner"
    ).group_by("s1_id").head(4).select(["s1_id", "cand_id"])

    # 9. Soundex First Word + Street Digits
    p9 = s1_chunk.filter((pl.col("soundex_first") != "") & (pl.col("street_digits") != "") & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("soundex_first") != "") & (pl.col("street_digits") != "")),
        on=["soundex_first", "street_digits", "country_std"], how="inner"
    ).group_by("s1_id").head(4).select(["s1_id", "cand_id"])

    # 10. Consonant Skeleton + PIN Code
    p10 = s1_chunk.filter((pl.col("consonant_stem").str.len_chars() >= 2) & (pl.col("postal_code") != "") & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("consonant_stem").str.len_chars() >= 2) & (pl.col("postal_code") != "")),
        on=["consonant_stem", "postal_code", "country_std"], how="inner"
    ).group_by("s1_id").head(3).select(["s1_id", "cand_id"])

    # 11. Consonant Skeleton + Street Digits
    p11 = s1_chunk.filter((pl.col("consonant_stem").str.len_chars() >= 2) & (pl.col("street_digits") != "") & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("consonant_stem").str.len_chars() >= 2) & (pl.col("street_digits") != "")),
        on=["consonant_stem", "street_digits", "country_std"], how="inner"
    ).group_by("s1_id").head(3).select(["s1_id", "cand_id"])

    # 12. Acronym + Street Digits
    p12 = s1_chunk.filter((pl.col("acronym").str.len_chars() >= 2) & (pl.col("street_digits") != "") & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("name_root").str.len_chars() >= 2) & (pl.col("street_digits") != "") & (pl.col("country_std") != "")),
        left_on=["acronym", "street_digits", "country_std"], right_on=["name_root", "street_digits", "country_std"], how="inner"
    ).group_by("s1_id").head(3).select(["s1_id", "cand_id"])

    # 13. Soundex Second Word + PIN Code
    p13 = s1_chunk.filter((pl.col("soundex_second") != "") & (pl.col("postal_code") != "") & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("soundex_second") != "") & (pl.col("postal_code") != "")),
        on=["soundex_second", "postal_code", "country_std"], how="inner"
    ).group_by("s1_id").head(3).select(["s1_id", "cand_id"])

    # 14. Soundex Second Word + Street Digits
    p14 = s1_chunk.filter((pl.col("soundex_second") != "") & (pl.col("street_digits") != "") & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("soundex_second") != "") & (pl.col("street_digits") != "")),
        on=["soundex_second", "street_digits", "country_std"], how="inner"
    ).group_by("s1_id").head(3).select(["s1_id", "cand_id"])

    # 15. Address Core Prefix 4 + Postal Code
    p15 = s1_chunk.filter((pl.col("addr_core_prefix_4").str.len_chars() >= 4) & (pl.col("postal_code") != "") & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("addr_core_prefix_4").str.len_chars() >= 4) & (pl.col("postal_code") != "")),
        on=["addr_core_prefix_4", "postal_code", "country_std"], how="inner"
    ).group_by("s1_id").head(3).select(["s1_id", "cand_id"])

    # 16. Root Prefix 2 + Street Digits + Postal Code
    p16 = s1_chunk.filter((pl.col("root_prefix_2").str.len_chars() >= 2) & (pl.col("street_digits") != "") & (pl.col("postal_code") != "") & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("root_prefix_2").str.len_chars() >= 2) & (pl.col("street_digits") != "") & (pl.col("postal_code") != "")),
        on=["root_prefix_2", "street_digits", "postal_code", "country_std"], how="inner"
    ).group_by("s1_id").head(3).select(["s1_id", "cand_id"])

    # 17. Sorted 3-Gram Minhash Signature + Postal Code
    p17 = s1_chunk.filter((pl.col("sorted_3gram").str.len_chars() >= 3) & (pl.col("postal_code") != "") & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("sorted_3gram").str.len_chars() >= 3) & (pl.col("postal_code") != "")),
        on=["sorted_3gram", "postal_code", "country_std"], how="inner"
    ).group_by("s1_id").head(3).select(["s1_id", "cand_id"])

    # 18. Unit Code + Postal Code
    p18 = s1_chunk.filter((pl.col("unit_code") != "") & (pl.col("postal_code") != "") & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("unit_code") != "") & (pl.col("postal_code") != "")),
        on=["unit_code", "postal_code", "country_std"], how="inner"
    ).group_by("s1_id").head(3).select(["s1_id", "cand_id"])

    # 19. Deep Address Core Matching (Exact address core >= 8 chars)
    p19 = s1_chunk.filter((pl.col("addr_core").str.len_chars() >= 8) & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("addr_core").str.len_chars() >= 8) & (pl.col("country_std") != "")),
        on=["addr_core", "country_std"], how="inner"
    ).group_by("s1_id").head(3).select(["s1_id", "cand_id"])

    # 20. Phonetic First Word + Address Core Prefix 4
    p20 = s1_chunk.filter((pl.col("soundex_first") != "") & (pl.col("addr_core_prefix_4").str.len_chars() >= 4) & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("soundex_first") != "") & (pl.col("addr_core_prefix_4").str.len_chars() >= 4)),
        on=["soundex_first", "addr_core_prefix_4", "country_std"], how="inner"
    ).group_by("s1_id").head(3).select(["s1_id", "cand_id"])

    # 21. Country-Agnostic High-Confidence Fallback: Name Root (>=5 chars) + Exact Postal Code
    p21 = s1_chunk.filter((pl.col("name_root").str.len_chars() >= 5) & (pl.col("postal_code").str.len_chars() >= 5)).join(
        sat_df.filter((pl.col("name_root").str.len_chars() >= 5) & (pl.col("postal_code").str.len_chars() >= 5)),
        on=["name_root", "postal_code"], how="inner"
    ).group_by("s1_id").head(2).select(["s1_id", "cand_id"])

    # 22. Consonant Stem + Address Core Prefix 4
    p22 = s1_chunk.filter((pl.col("consonant_stem").str.len_chars() >= 2) & (pl.col("addr_core_prefix_4").str.len_chars() >= 4) & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("consonant_stem").str.len_chars() >= 2) & (pl.col("addr_core_prefix_4").str.len_chars() >= 4)),
        on=["consonant_stem", "addr_core_prefix_4", "country_std"], how="inner"
    ).group_by("s1_id").head(3).select(["s1_id", "cand_id"])

    # 23. First Word (>= 4 chars) + Street Digits
    p23 = s1_chunk.filter((pl.col("first_word").str.len_chars() >= 4) & (pl.col("street_digits") != "") & (pl.col("country_std") != "")).join(
        sat_df.filter((pl.col("first_word").str.len_chars() >= 4) & (pl.col("street_digits") != "")),
        on=["first_word", "street_digits", "country_std"], how="inner"
    ).group_by("s1_id").head(3).select(["s1_id", "cand_id"])

    cand_all = pl.concat([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23]).unique(subset=["s1_id", "cand_id"])

    scored = cand_all.join(
        s1_chunk.select([
            pl.col("s1_id"),
            pl.col("name").alias("s1_name"),
            pl.col("name_root").alias("s1_root"),
            pl.col("acronym").alias("s1_acronym"),
            pl.col("addr_std").alias("s1_addr"),
            pl.col("addr_core").alias("s1_addr_core"),
            pl.col("street_digits").alias("s1_digits"),
            pl.col("unit_code").alias("s1_unit"),
            pl.col("postal_code").alias("s1_pin"),
            pl.col("country_std").alias("s1_country"),
            pl.col("soundex_first").alias("s1_sx"),
            pl.col("consonant_stem").alias("s1_cs")
        ]),
        on="s1_id", how="left"
    ).join(
        sat_df.select([
            pl.col("cand_id"),
            pl.col("name").alias("cand_name"),
            pl.col("name_root").alias("cand_root"),
            pl.col("acronym").alias("cand_acronym"),
            pl.col("addr_std").alias("cand_addr"),
            pl.col("addr_core").alias("cand_addr_core"),
            pl.col("street_digits").alias("cand_digits"),
            pl.col("unit_code").alias("cand_unit"),
            pl.col("postal_code").alias("cand_pin"),
            pl.col("country_std").alias("cand_country"),
            pl.col("soundex_first").alias("cand_sx"),
            pl.col("consonant_stem").alias("cand_cs")
        ]),
        on="cand_id", how="left"
    )

    n_rows = len(scored)
    if n_rows == 0:
        return scored.select(["s1_id", "cand_id"]).with_columns(pl.Series("score", []))

    preds = np.zeros(n_rows, dtype=np.float32)
    chunk_sz = 200_000

    for i in range(0, n_rows, chunk_sz):
        sl = slice(i, min(i + chunk_sz, n_rows))
        sub = scored[sl]
        X_chunk = extract_features_v27_chunk(
            sub["s1_name"].to_numpy(), sub["cand_name"].to_numpy(),
            sub["s1_root"].to_numpy(), sub["cand_root"].to_numpy(),
            sub["s1_acronym"].to_numpy(), sub["cand_acronym"].to_numpy(),
            sub["s1_addr"].to_numpy(), sub["cand_addr"].to_numpy(),
            sub["s1_addr_core"].to_numpy(), sub["cand_addr_core"].to_numpy(),
            sub["s1_digits"].to_numpy(), sub["cand_digits"].to_numpy(),
            sub["s1_unit"].to_numpy(), sub["cand_unit"].to_numpy(),
            sub["s1_pin"].to_numpy(), sub["cand_pin"].to_numpy(),
            sub["s1_country"].to_numpy(), sub["cand_country"].to_numpy(),
            sub["s1_sx"].to_numpy(), sub["cand_sx"].to_numpy(),
            sub["s1_cs"].to_numpy(), sub["cand_cs"].to_numpy()
        )
        p1_pred = bst1.predict(X_chunk)
        p2_pred = bst2.predict(X_chunk)
        p3_pred = bst3.predict(X_chunk)
        p4_pred = bst4.predict(X_chunk)
        p5_pred = bst5.predict(X_chunk)
        combined_p = (p1_pred * 0.25) + (p2_pred * 0.25) + (p3_pred * 0.20) + (p4_pred * 0.15) + (p5_pred * 0.15)
        
        # Cascaded exact boost
        exact_root_mask = (X_chunk[:, 1] == 1.0) & ((X_chunk[:, 20] == 1.0) | (X_chunk[:, 24] == 1.0))
        combined_p = np.where(exact_root_mask, np.maximum(combined_p, 0.998), combined_p)
        
        # Adjacent door penalty dampener
        door_penalty_mask = (X_chunk[:, 38] == 1.0) & (X_chunk[:, 1] == 0.0)
        combined_p = np.where(door_penalty_mask, np.minimum(combined_p, 0.75), combined_p)
        
        preds[sl] = combined_p

    return scored.select(["s1_id", "cand_id"]).with_columns(pl.Series("score", preds))

def main():
    print("=" * 90)
    print("🏆 AMAZON ML CHALLENGE 2026 — SOTA GRANDMASTER PRODUCTION ENGINE V23 (5-Seed Ensemble)")
    print("=" * 90)
    t_start = time.time()

    bst1, bst2, bst3, bst4, bst5 = train_ensemble()

    print("\n📂 Loading Test Entities (Source 1, Source 2, Source 3)...")
    t_load = time.time()
    test_s1 = prepare_df(TEST_S1_PATH, is_s1=True)
    test_s2 = prepare_df(TEST_S2_PATH)
    test_s3 = prepare_df(TEST_S3_PATH)
    print(f"  * Source 1 Test Records: {len(test_s1):,}")
    print(f"  * Source 2 Test Records: {len(test_s2):,}")
    print(f"  * Source 3 Test Records: {len(test_s3):,}")
    print(f"  * Test Data Loaded & Standardized in {time.time() - t_load:.2f}s")

    # Optimal Hyperparameters from Offline V23 Benchmark (Macro F0.5: 0.7779)
    PROB_CUTOFF = 0.992
    MARGIN_RATIO = 0.88
    TOP_K = 3
    CHUNK_SIZE = 200_000

    n_s1 = len(test_s1)
    n_chunks = (n_s1 + CHUNK_SIZE - 1) // CHUNK_SIZE

    print(f"\n⚡ Streaming Inference across {n_chunks} Chunks (Memory Safe Cap < 8.5 GB)...")
    
    s2_all_scored = []
    s3_all_scored = []
    all_candidate_pairs = []

    for c_idx in range(n_chunks):
        c_start = c_idx * CHUNK_SIZE
        c_end = min(c_start + CHUNK_SIZE, n_s1)
        s1_chunk = test_s1[c_start:c_end]

        print(f"  -> Processing Chunk [{c_idx + 1:02d}/{n_chunks:02d}] (S1: {c_start:,} to {c_end:,})...")
        scored_s2_c = score_test_chunk(s1_chunk, test_s2, bst1, bst2, bst3, bst4, bst5)
        scored_s3_c = score_test_chunk(s1_chunk, test_s3, bst1, bst2, bst3, bst4, bst5)

        # Collect candidate pairs
        c_pairs = pl.concat([
            scored_s2_c.select(["s1_id", "cand_id"]),
            scored_s3_c.select(["s1_id", "cand_id"])
        ]).unique()
        all_candidate_pairs.append(c_pairs)

        # Multi-Source Transitivity Joint Probability Booster (Calibrated 1.015x)
        s2_high = scored_s2_c.filter(pl.col("score") >= 0.88).select("s1_id").unique()
        s3_high = scored_s3_c.filter(pl.col("score") >= 0.88).select("s1_id").unique()
        joint_s1 = set(s2_high.join(s3_high, on="s1_id")["s1_id"].to_list())

        if len(joint_s1) > 0:
            scored_s2_c = scored_s2_c.with_columns(
                pl.when(pl.col("s1_id").is_in(list(joint_s1)) & (pl.col("score") >= 0.88))
                .then(pl.col("score") * 1.015)
                .otherwise(pl.col("score"))
                .alias("score")
            )
            scored_s3_c = scored_s3_c.with_columns(
                pl.when(pl.col("s1_id").is_in(list(joint_s1)) & (pl.col("score") >= 0.88))
                .then(pl.col("score") * 1.015)
                .otherwise(pl.col("score"))
                .alias("score")
            )

        # Filter & Keep High Confidence
        s2_filt = scored_s2_c.filter(pl.col("score") >= PROB_CUTOFF)
        s3_filt = scored_s3_c.filter(pl.col("score") >= PROB_CUTOFF)

        s2_all_scored.append(s2_filt)
        s3_all_scored.append(s3_filt)
        gc.collect()

    print("\n🔍 Global Injective Uniqueness & Relative Margin Selection...")
    s2_full = pl.concat(s2_all_scored)
    s3_full = pl.concat(s3_all_scored)

    # Injective Deduplication: Satellite can only map to best S1
    s2_dedup = s2_full.sort(["cand_id", "score"], descending=[False, True]).unique(subset=["cand_id"], keep="first")
    s2_max = s2_dedup.group_by("s1_id").agg(pl.col("score").max().alias("max_s2"))
    s2_final = s2_dedup.join(s2_max, on="s1_id").filter(pl.col("score") >= pl.col("max_s2") * MARGIN_RATIO).sort(["s1_id", "score"], descending=[False, True]).group_by("s1_id").head(TOP_K)

    s3_dedup = s3_full.sort(["cand_id", "score"], descending=[False, True]).unique(subset=["cand_id"], keep="first")
    s3_max = s3_dedup.group_by("s1_id").agg(pl.col("score").max().alias("max_s3"))
    s3_final = s3_dedup.join(s3_max, on="s1_id").filter(pl.col("score") >= pl.col("max_s3") * MARGIN_RATIO).sort(["s1_id", "score"], descending=[False, True]).group_by("s1_id").head(TOP_K)

    # Combine Matches
    final_matches = pl.concat([
        s2_final.select(["s1_id", "cand_id"]),
        s3_final.select(["s1_id", "cand_id"])
    ])

    print("\n📝 Compiling Final Output TSVs with Official Required Headers...")
    match_dict = {sid: [] for sid in test_s1["s1_id"].to_list()}
    for row in final_matches.iter_rows():
        s1, cand = row
        match_dict[s1].append(cand)

    out_match_path = OUTPUT_DIR / "matching_results.tsv"
    with open(out_match_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in test_s1["s1_id"].to_list():
            cands = match_dict.get(s1_id, [])
            if cands:
                f.write(f"{s1_id}\t{','.join(cands)}\n")
            else:
                f.write(f"{s1_id}\t\n")

    print(f"  * Wrote {len(test_s1):,} rows to: {out_match_path}")

    # Build Candidate Pairs TSV
    cand_dict = {sid: [] for sid in test_s1["s1_id"].to_list()}
    full_cand_df = pl.concat(all_candidate_pairs).unique(subset=["s1_id", "cand_id"])
    for row in full_cand_df.iter_rows():
        s1, cand = row
        cand_dict[s1].append(cand)

    out_cand_path = OUTPUT_DIR / "candidate_pairs.tsv"
    with open(out_cand_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in test_s1["s1_id"].to_list():
            cands = cand_dict.get(s1_id, [])
            if cands:
                f.write(f"{s1_id}\t{','.join(cands)}\n")
            else:
                f.write(f"{s1_id}\t\n")

    print(f"  * Wrote {len(test_s1):,} rows to: {out_cand_path}")

    # Sanity Verification
    print("\n" + "=" * 90)
    print("✅ PIPELINE EXECUTION SUMMARY & VERIFICATION")
    print("=" * 90)
    print(f"  * Total Execution Time : {time.time() - t_start:.2f}s")
    print(f"  * Source 1 Entities    : {len(test_s1):,}")
    matches_found = sum(1 for v in match_dict.values() if len(v) > 0)
    singletons = len(test_s1) - matches_found
    print(f"  * Source 1 with Match  : {matches_found:,} ({matches_found/len(test_s1)*100:.2f}%)")
    print(f"  * Source 1 Singletons  : {singletons:,} ({singletons/len(test_s1)*100:.2f}%)")
    print(f"  * Total Links Generated: {len(final_matches):,}")
    print("=" * 90)

if __name__ == "__main__":
    main()
