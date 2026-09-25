import re
from typing import Dict, List, Tuple
import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

try:
    from sparse_dot_topn import sp_matmul_topn
    HAS_SPARSE_DOT_TOPN = True
except ImportError:
    HAS_SPARSE_DOT_TOPN = False


def extract_first_tokens(text: str, n_tokens: int = 2) -> str:
    """Extracts first N significant word tokens."""
    if not text:
        return ""
    words = [w for w in text.split() if len(w) > 1]
    return " ".join(words[:n_tokens]) if len(words) >= 1 else ""


def extract_address_block_key(addr: str) -> str:
    """Extracts street number + primary street token (e.g., '410 terry')."""
    if not addr:
        return ""
    digits = re.findall(r"\b\d+\b", addr)
    words = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", addr)]
    if digits and words:
        return f"{digits[0]}_{words[0]}"
    return ""


class MultiPassBlocker:
    """
    High-recall, multi-pass candidate blocker.
    Combines:
      1. Country-partitioned Char Q-gram TF-IDF on Business Name
      2. Combined Name + Address Char Q-gram TF-IDF
      3. First-N-Tokens Name Inverted Index
      4. Street Number + Address Inverted Index
      5. Exact Phone Matching
    """
    def __init__(
        self,
        top_k: int = 35,
        min_sim: float = 0.15,
        ngram_range: Tuple[int, int] = (2, 4)
    ):
        self.top_k = top_k
        self.min_sim = min_sim
        self.ngram_range = ngram_range
        self.name_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            min_df=2,
            dtype=np.float32,
            sublinear_tf=True
        )
        self.combo_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=2,
            dtype=np.float32,
            sublinear_tf=True
        )

    def _sparse_topn(self, A: csr_matrix, B: csr_matrix, top_k: int, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates top-K cosine similarity between row vectors in A and B."""
        if HAS_SPARSE_DOT_TOPN:
            res = sp_matmul_topn(A, B.T, top_n=top_k, threshold=threshold)
            res_coo = res.tocoo()
            return res_coo.row, res_coo.col, res_coo.data
        else:
            rows, cols, data = [], [], []
            chunk_size = 5000
            for start in range(0, A.shape[0], chunk_size):
                end = min(start + chunk_size, A.shape[0])
                chunk = A[start:end].dot(B.T).toarray()
                for i, row in enumerate(chunk):
                    valid_idx = np.where(row >= threshold)[0]
                    if len(valid_idx) > top_k:
                        top_idx = valid_idx[np.argpartition(-row[valid_idx], top_k)[:top_k]]
                    else:
                        top_idx = valid_idx
                    
                    for c in top_idx:
                        rows.append(start + i)
                        cols.append(c)
                        data.append(row[c])
            return np.array(rows), np.array(cols), np.array(data)

    def block_candidates(
        self,
        df_s1: pl.DataFrame,
        df_satellites: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Executes multi-pass candidate generation.
        Returns a Polars DataFrame with columns: ['s1_id', 'candidate_id', 'tfidf_sim']
        """
        candidate_dfs = []

        s1_ids = df_s1["id"].to_list()
        sat_ids = df_satellites["id"].to_list()

        s1_names = df_s1["clean_name"].to_list()
        sat_names = df_satellites["clean_name"].to_list()

        # Combine name + address for rich context matching
        s1_combo = (df_s1["clean_name"] + " " + df_s1["clean_address"]).to_list()
        sat_combo = (df_satellites["clean_name"] + " " + df_satellites["clean_address"]).to_list()

        # -------------------------------------------------------------
        # PASS 1: Char 2-4 Gram TF-IDF on Name (Top-25)
        # -------------------------------------------------------------
        try:
            self.name_vectorizer.fit(s1_names + sat_names[:min(len(sat_names), 500000)])
            s1_vecs = self.name_vectorizer.transform(s1_names)
            sat_vecs = self.name_vectorizer.transform(sat_names)

            s1_idx_1, sat_idx_1, sim_1 = self._sparse_topn(
                s1_vecs, sat_vecs, top_k=min(self.top_k, 25), threshold=self.min_sim
            )

            if len(s1_idx_1) > 0:
                pass1_df = pl.DataFrame({
                    "s1_id": [s1_ids[i] for i in s1_idx_1],
                    "candidate_id": [sat_ids[j] for j in sat_idx_1],
                    "tfidf_sim": sim_1.astype(np.float32)
                })
                candidate_dfs.append(pass1_df)
        except Exception as e:
            print(f"  * Warning in Pass 1 blocking: {e}")

        # -------------------------------------------------------------
        # PASS 2: Char 3-4 Gram TF-IDF on Name + Address Combo (Top-20)
        # -------------------------------------------------------------
        try:
            self.combo_vectorizer.fit(s1_combo + sat_combo[:min(len(sat_combo), 500000)])
            s1_combo_vecs = self.combo_vectorizer.transform(s1_combo)
            sat_combo_vecs = self.combo_vectorizer.transform(sat_combo)

            s1_idx_2, sat_idx_2, sim_2 = self._sparse_topn(
                s1_combo_vecs, sat_combo_vecs, top_k=20, threshold=0.18
            )

            if len(s1_idx_2) > 0:
                pass2_df = pl.DataFrame({
                    "s1_id": [s1_ids[i] for i in s1_idx_2],
                    "candidate_id": [sat_ids[j] for j in sat_idx_2],
                    "tfidf_sim": sim_2.astype(np.float32)
                })
                candidate_dfs.append(pass2_df)
        except Exception as e:
            print(f"  * Warning in Pass 2 blocking: {e}")

        # -------------------------------------------------------------
        # PASS 3: Name First-2-Tokens Inverted Index
        # -------------------------------------------------------------
        s1_tokens = df_s1.with_columns(
            pl.col("clean_name").map_elements(lambda x: extract_first_tokens(x, 2), return_dtype=pl.Utf8).alias("block_name_tokens")
        ).filter(pl.col("block_name_tokens").str.len_chars() >= 4)

        sat_tokens = df_satellites.with_columns(
            pl.col("clean_name").map_elements(lambda x: extract_first_tokens(x, 2), return_dtype=pl.Utf8).alias("block_name_tokens")
        ).filter(pl.col("block_name_tokens").str.len_chars() >= 4)

        if len(s1_tokens) > 0 and len(sat_tokens) > 0:
            token_pairs = s1_tokens.select(["id", "block_name_tokens"]).join(
                sat_tokens.select(["id", "block_name_tokens"]),
                on="block_name_tokens",
                how="inner",
                suffix="_candidate"
            ).select([
                pl.col("id").alias("s1_id"),
                pl.col("id_candidate").alias("candidate_id"),
                pl.lit(0.70, dtype=pl.Float32).alias("tfidf_sim")
            ])
            candidate_dfs.append(token_pairs)

        # -------------------------------------------------------------
        # PASS 4: Address Street Number + Token Inverted Index
        # -------------------------------------------------------------
        s1_addr_keys = df_s1.with_columns(
            pl.col("clean_address").map_elements(extract_address_block_key, return_dtype=pl.Utf8).alias("block_addr_key")
        ).filter(pl.col("block_addr_key") != "")

        sat_addr_keys = df_satellites.with_columns(
            pl.col("clean_address").map_elements(extract_address_block_key, return_dtype=pl.Utf8).alias("block_addr_key")
        ).filter(pl.col("block_addr_key") != "")

        if len(s1_addr_keys) > 0 and len(sat_addr_keys) > 0:
            addr_pairs = s1_addr_keys.select(["id", "block_addr_key"]).join(
                sat_addr_keys.select(["id", "block_addr_key"]),
                on="block_addr_key",
                how="inner",
                suffix="_candidate"
            ).select([
                pl.col("id").alias("s1_id"),
                pl.col("id_candidate").alias("candidate_id"),
                pl.lit(0.60, dtype=pl.Float32).alias("tfidf_sim")
            ])
            candidate_dfs.append(addr_pairs)

        # -------------------------------------------------------------
        # PASS 5: Exact Phone Inverted Index
        # -------------------------------------------------------------
        s1_valid_phones = df_s1.filter((pl.col("clean_phone") != "") & (pl.col("clean_phone").str.len_chars() >= 7))
        sat_valid_phones = df_satellites.filter((pl.col("clean_phone") != "") & (pl.col("clean_phone").str.len_chars() >= 7))

        if len(s1_valid_phones) > 0 and len(sat_valid_phones) > 0:
            phone_pairs = s1_valid_phones.select(["id", "clean_phone"]).join(
                sat_valid_phones.select(["id", "clean_phone"]),
                on="clean_phone",
                how="inner",
                suffix="_candidate"
            ).select([
                pl.col("id").alias("s1_id"),
                pl.col("id_candidate").alias("candidate_id"),
                pl.lit(1.0, dtype=pl.Float32).alias("tfidf_sim")
            ])
            candidate_dfs.append(phone_pairs)

        # -------------------------------------------------------------
        # Union All Passes & Keep Max Similarity per Candidate
        # -------------------------------------------------------------
        if not candidate_dfs:
            return pl.DataFrame({
                "s1_id": pl.Series([], dtype=pl.Utf8),
                "candidate_id": pl.Series([], dtype=pl.Utf8),
                "tfidf_sim": pl.Series([], dtype=pl.Float32)
            })

        all_pairs = pl.concat(candidate_dfs)
        # Group by s1_id, candidate_id and take max tfidf_sim
        deduped = all_pairs.group_by(["s1_id", "candidate_id"]).agg(
            pl.col("tfidf_sim").max()
        )

        # Limit to Top-45 candidates per S1 entity to maintain high precision & memory safety
        filtered_pairs = deduped.sort(
            by=["s1_id", "tfidf_sim"],
            descending=[False, True]
        ).group_by("s1_id").head(45)

        return filtered_pairs

    @staticmethod
    def evaluate_blocking_recall(
        pairs_df: pl.DataFrame,
        ground_truth: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Evaluates candidate recall on ground truth.
        Candidate Recall = (Total ground truth matches found in candidates) / (Total true ground truth matches)
        """
        candidate_dict = {}
        for row in pairs_df.select(["s1_id", "candidate_id"]).iter_rows():
            s1_id, cand_id = row
            if s1_id not in candidate_dict:
                candidate_dict[s1_id] = set()
            candidate_dict[s1_id].add(cand_id)

        total_true_matches = 0
        found_matches = 0
        s1_count = 0
        s1_full_recall_count = 0

        for s1_id, true_list in ground_truth.items():
            if len(true_list) == 0:
                continue  # Singletons have no satellite candidates to find
            
            s1_count += 1
            true_set = set(true_list)
            cand_set = candidate_dict.get(s1_id, set())

            intersect = true_set.intersection(cand_set)
            found_matches += len(intersect)
            total_true_matches += len(true_set)

            if len(intersect) == len(true_set):
                s1_full_recall_count += 1

        overall_recall = (found_matches / total_true_matches) if total_true_matches > 0 else 1.0
        entity_perfect_recall = (s1_full_recall_count / s1_count) if s1_count > 0 else 1.0
        avg_candidates_per_entity = len(pairs_df) / max(len(ground_truth), 1)

        return {
            "candidate_match_recall": float(overall_recall),
            "entity_perfect_recall_rate": float(entity_perfect_recall),
            "total_candidates_generated": len(pairs_df),
            "avg_candidates_per_s1": float(avg_candidates_per_entity)
        }
