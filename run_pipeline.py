import argparse
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.pipeline import EntityResolutionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026: Business Entity Resolution Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cv", "submit", "both"],
        default="cv",
        help="Execution mode: 'cv' for local validation, 'submit' for generating test submission, 'both' for CV + submit."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=25000,
        help="Number of S1 entities to sample for fast local iteration (default: 25,000). Set --full to run on complete dataset."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run on 100% of the training dataset (overrides --sample-size)."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=35,
        help="Top-K candidates per S1 entity generated in blocking step (default: 35)."
    )
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.25,
        help="Minimum TF-IDF cosine similarity threshold for candidate generation (default: 0.25)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output path for submission CSV."
    )

    args = parser.parse_args()

    pipeline = EntityResolutionPipeline(
        top_k_candidates=args.top_k,
        min_blocking_sim=args.min_sim
    )

    sample_size = None if args.full else args.sample_size

    if args.mode in ["cv", "both"]:
        print(f"\n[INFO] Starting Local Cross-Validation (sample_size={sample_size if sample_size else 'FULL'})...")
        pipeline.run_cv_evaluation(sample_size=sample_size)

    if args.mode in ["submit", "both"]:
        print(f"\n[INFO] Generating Submission on Test Set...")
        out_path = Path(args.output) if args.output else None
        pipeline.run_submission(output_path=out_path)


if __name__ == "__main__":
    main()
