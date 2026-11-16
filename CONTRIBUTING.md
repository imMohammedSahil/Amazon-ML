# Contributing to Amazon ML Challenge — Entity Resolution

Thank you for your interest in contributing to the **Business Entity Resolution** project. Whether you're reporting edge-case anomalies, proposing algorithmic improvements, or adding unit tests, all contributions are welcome.

---

## Code of Conduct

Everyone participating in this project is expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md). Please report any violations following the enforcement guidelines.

---

## How to Contribute

### 1. Reporting Bugs & Inconsistencies

When reporting issues with blocking logic, feature extraction, or inference scripts:
* Describe the bug or edge case with minimal reproducer rows (e.g. specific source entity triplets).
* Include logs or tracebacks (especially from Polars/LightGBM).
* State your Python version and environment dependencies (`pip list` / `requirements.txt`).

### 2. Proposing Model or Blocking Improvements

If you are contributing novel blocking passes or feature definitions:
* Ensure all transforms operate cleanly on Polars/NumPy vectorized routines.
* Validate memory footprint and runtime complexity—full datasets contain 1.7M+ entities and require strict memory discipline.
* Benchmark offline cross-validation impact using `python run_pipeline.py --mode cv`.

### 3. Submitting Pull Requests

1. **Fork the repository** on GitHub.
2. **Clone your fork**:
   ```bash
   git clone https://github.com/<your-username>/Amazon-ML.git
   cd Amazon-ML
   ```
3. **Create a topic branch**:
   ```bash
   git checkout -b feature/blocking-pass-optimization
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Run test suite**:
   ```bash
   pytest tests/ -v
   ```
6. **Commit and push**:
   ```bash
   git commit -m "feat(blocking): add vectorized phonetic stem join pass"
   git push origin feature/blocking-pass-optimization
   ```
7. **Open a Pull Request** against the `develop` branch describing your changes and holdout evaluation metrics.

---

## License

By contributing to this repository, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
