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


class MultiPassBlocker:
    """
    High-recall, multi-pass candidate blocker.
    Generates candidate match pairs between anchor records (S1) and satellite records (S2 + S3).
    """
    def __init__(
        self,
        top_k: int = 35,
        min_sim: float = 0.25,
        ngram_range: Tuple[int, int] = (3, 4)
    ):
        self.top_k = top_k
        self.min_sim = min_sim
        self.ngram_range = ngram_range
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            min_df=2,
            dtype=np.float32
        )

    def _sparse_topn(self, A: csr_matrix, B: csr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates top-K cosine similarity between row vectors in A and B."""
        if HAS_SPARSE_DOT_TOPN:
            # Ultra-fast C++ top-K sparse matrix product
            res = sp_matmul_topn(A, B.T, top_n=self.top_k, threshold=self.min_sim)
            # res is a scipy CSR matrix with top_k items per row
            res_coo = res.tocoo()
            return res_coo.row, res_coo.col, res_coo.data
        else:
            # Fallback pure SciPy chunked implementation
            rows, cols, data = [], [], []
            chunk_size = 5000
            for start in range(0, A.shape[0], chunk_size):
                end = min(start + chunk_size, A.shape[0])
                chunk = A[start:end].dot(B.T).toarray()
                for i, row in enumerate(chunk):
                    # Get indices with similarity >= min_sim
                    valid_idx = np.where(row >= self.min_sim)[0]
                    if len(valid_idx) > self.top_k:
                        # Top-K
                        top_idx = valid_idx[np.argpartition(-row[valid_idx], self.top_k)[:self.top_k]]
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
        Returns a Polars DataFrame with columns: ['s1_id', 'candidate_id', 'tfidf_sim', 'exact_phone_match']
        """
        s1_ids = df_s1["id"].to_list()
        sat_ids = df_satellites["id"].to_list()

        s1_names = df_s1["clean_name"].to_list()
        sat_names = df_satellites["clean_name"].to_list()

        # Pass 1: TF-IDF Character Q-Gram Cosine Similarity
        # Fit vectorizer on union of all unique names
        all_names = list(set(s1_names + sat_names))
        self.tfidf_vectorizer.fit(all_names)

        s1_vecs = self.tfidf_vectorizer.transform(s1_names)
        sat_vecs = self.tfidf_vectorizer.transform(sat_names)

        s1_idx_arr, sat_idx_arr, sim_arr = self._sparse_topn(s1_vecs, sat_vecs)

        # Map integer indices back to entity IDs
        s1_matched_ids = [s1_ids[i] for i in s1_idx_arr]
        sat_matched_ids = [sat_ids[j] for j in sat_idx_arr]

        pairs_df = pl.DataFrame({
            "s1_id": s1_matched_ids,
            "candidate_id": sat_matched_ids,
            "tfidf_sim": sim_arr.astype(np.float32)
        })

        # Pass 2: Exact Phone Match Blocking (catches entities with misspelled names but identical phone)
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
            
            # Combine pairs and take max similarity for duplicates
            pairs_df = pl.concat([pairs_df, phone_pairs]).unique(subset=["s1_id", "candidate_id"], keep="first")

        return pairs_df

    @staticmethod
    def evaluate_blocking_recall(
        pairs_df: pl.DataFrame,
        ground_truth: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Evaluates candidate recall on ground truth.
        Candidate Recall = (Total ground truth matches found in candidates) / (Total true ground truth matches)
        """
        # Group candidates by s1_id
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
