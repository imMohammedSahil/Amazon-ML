import re
from typing import List, Set
import polars as pl
import numpy as np
from rapidfuzz import fuzz, distance


def extract_street_digits(address_str: str) -> Set[str]:
    """Extracts numeric street numbers."""
    if not address_str:
        return set()
    return set(re.findall(r"\b\d+\b", address_str))


def compute_word_jaccard(s1: str, s2: str) -> float:
    """Computes word-level Jaccard similarity."""
    if not s1 or not s2:
        return 0.0
    w1 = set(s1.split())
    w2 = set(s2.split())
    if not w1 or not w2:
        return 0.0
    return float(len(w1.intersection(w2)) / len(w1.union(w2)))


class PairwiseFeatureExtractor:
    """
    Extracts deep pairwise similarity features for candidate pairs.
    Operates on joined Polars tables for maximum throughput and vectorized execution.
    """
    def __init__(self):
        self.feature_names: List[str] = []

    def extract_features(
        self,
        pairs_df: pl.DataFrame,
        df_s1: pl.DataFrame,
        df_satellites: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Joins candidate pairs with S1 and Satellite metadata, then computes all pairwise features.
        Returns a DataFrame containing:
          - ID columns: ['s1_id', 'candidate_id']
          - Feature columns: ['feat_name_ratio', 'feat_name_token_set', 'feat_name_partial', 'feat_name_jw', ...]
        """
        # Join metadata for S1
        s1_cols = [
            pl.col("id").alias("s1_id"),
            pl.col("clean_name").alias("s1_name"),
            pl.col("clean_name_raw").alias("s1_name_raw"),
            pl.col("clean_address").alias("s1_address"),
            pl.col("clean_city").alias("s1_city"),
            pl.col("clean_state").alias("s1_state"),
            pl.col("clean_zip").alias("s1_zip"),
            pl.col("clean_country").alias("s1_country"),
            pl.col("clean_phone").alias("s1_phone"),
            pl.col("clean_full_address").alias("s1_full_address")
        ]
        
        # Join metadata for Satellites
        sat_cols = [
            pl.col("id").alias("candidate_id"),
            pl.col("source").alias("cand_source"),
            pl.col("clean_name").alias("cand_name"),
            pl.col("clean_name_raw").alias("cand_name_raw"),
            pl.col("clean_address").alias("cand_address"),
            pl.col("clean_city").alias("cand_city"),
            pl.col("clean_state").alias("cand_state"),
            pl.col("clean_zip").alias("cand_zip"),
            pl.col("clean_country").alias("cand_country"),
            pl.col("clean_phone").alias("cand_phone"),
            pl.col("clean_full_address").alias("cand_full_address")
        ]

        # Join S1 and Satellite data onto pairs
        joined = pairs_df.join(
            df_s1.select(s1_cols), on="s1_id", how="left"
        ).join(
            df_satellites.select(sat_cols), on="candidate_id", how="left"
        )

        s1_names = joined["s1_name"].to_list()
        cand_names = joined["cand_name"].to_list()
        s1_addrs = joined["s1_full_address"].to_list()
        cand_addrs = joined["cand_full_address"].to_list()

        n_pairs = len(joined)

        # Preallocate numpy arrays for fast features
        feat_name_ratio = np.zeros(n_pairs, dtype=np.float32)
        feat_name_token_set = np.zeros(n_pairs, dtype=np.float32)
        feat_name_token_sort = np.zeros(n_pairs, dtype=np.float32)
        feat_name_partial = np.zeros(n_pairs, dtype=np.float32)
        feat_name_jw = np.zeros(n_pairs, dtype=np.float32)
        feat_name_jaccard = np.zeros(n_pairs, dtype=np.float32)
        feat_name_len_ratio = np.zeros(n_pairs, dtype=np.float32)
        feat_first_token_match = np.zeros(n_pairs, dtype=np.float32)
        
        feat_addr_ratio = np.zeros(n_pairs, dtype=np.float32)
        feat_addr_token_set = np.zeros(n_pairs, dtype=np.float32)
        feat_addr_jw = np.zeros(n_pairs, dtype=np.float32)
        feat_street_num_match = np.zeros(n_pairs, dtype=np.float32)

        for i in range(n_pairs):
            n1 = s1_names[i]
            n2 = cand_names[i]
            
            if n1 and n2:
                feat_name_ratio[i] = fuzz.ratio(n1, n2) / 100.0
                feat_name_token_set[i] = fuzz.token_set_ratio(n1, n2) / 100.0
                feat_name_token_sort[i] = fuzz.token_sort_ratio(n1, n2) / 100.0
                feat_name_partial[i] = fuzz.partial_ratio(n1, n2) / 100.0
                feat_name_jw[i] = distance.JaroWinkler.similarity(n1, n2)
                feat_name_jaccard[i] = compute_word_jaccard(n1, n2)
                
                l1, l2 = len(n1), len(n2)
                feat_name_len_ratio[i] = min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 1.0

                # First token match
                t1 = n1.split()[0] if n1 else ""
                t2 = n2.split()[0] if n2 else ""
                feat_first_token_match[i] = 1.0 if (t1 and t2 and t1 == t2) else 0.0

            a1 = s1_addrs[i]
            a2 = cand_addrs[i]
            if a1 and a2:
                feat_addr_ratio[i] = fuzz.ratio(a1, a2) / 100.0
                feat_addr_token_set[i] = fuzz.token_set_ratio(a1, a2) / 100.0
                feat_addr_jw[i] = distance.JaroWinkler.similarity(a1, a2)
                
                # Check if street numbers match
                d1 = extract_street_digits(a1)
                d2 = extract_street_digits(a2)
                if d1 and d2 and d1.intersection(d2):
                    feat_street_num_match[i] = 1.0

        # Exact and Match Features via Polars columnar expressions
        feature_df = joined.with_columns([
            pl.Series("feat_name_ratio", feat_name_ratio),
            pl.Series("feat_name_token_set", feat_name_token_set),
            pl.Series("feat_name_token_sort", feat_name_token_sort),
            pl.Series("feat_name_partial", feat_name_partial),
            pl.Series("feat_name_jw", feat_name_jw),
            pl.Series("feat_name_jaccard", feat_name_jaccard),
            pl.Series("feat_name_len_ratio", feat_name_len_ratio),
            pl.Series("feat_first_token_match", feat_first_token_match),
            pl.Series("feat_addr_ratio", feat_addr_ratio),
            pl.Series("feat_addr_token_set", feat_addr_token_set),
            pl.Series("feat_addr_jw", feat_addr_jw),
            pl.Series("feat_street_num_match", feat_street_num_match),
            
            # Name exact match flag
            (pl.col("s1_name") == pl.col("cand_name")).cast(pl.Float32).alias("feat_name_exact"),
            
            # City / State / Country matches
            ((pl.col("s1_city") != "") & (pl.col("s1_city") == pl.col("cand_city"))).cast(pl.Float32).alias("feat_city_exact"),
            ((pl.col("s1_state") != "") & (pl.col("s1_state") == pl.col("cand_state"))).cast(pl.Float32).alias("feat_state_exact"),
            ((pl.col("s1_country") != "") & (pl.col("s1_country") == pl.col("cand_country"))).cast(pl.Float32).alias("feat_country_exact"),
            
            # Country Mismatch penalty indicator (when both present and different)
            ((pl.col("s1_country") != "") & (pl.col("cand_country") != "") & (pl.col("s1_country") != pl.col("cand_country"))).cast(pl.Float32).alias("feat_country_mismatch"),
            
            # Zip match
            ((pl.col("s1_zip") != "") & (pl.col("s1_zip") == pl.col("cand_zip"))).cast(pl.Float32).alias("feat_zip_exact"),
            
            # Phone matches
            ((pl.col("s1_phone") != "") & (pl.col("s1_phone") == pl.col("cand_phone"))).cast(pl.Float32).alias("feat_phone_exact"),
            ((pl.col("s1_phone") == "") | (pl.col("cand_phone") == "")).cast(pl.Float32).alias("feat_phone_missing"),
            
            # Source indicator (is candidate from S2 or S3)
            (pl.col("cand_source") == "source2").cast(pl.Float32).alias("feat_is_source2"),
            
            # Blocking cosine similarity
            pl.col("tfidf_sim").fill_null(0.0).alias("feat_blocking_tfidf_sim")
        ])

        # Track all generated feature names
        self.feature_names = [col for col in feature_df.columns if col.startswith("feat_")]
        return feature_df
