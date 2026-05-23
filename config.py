"""
Central configuration for SentiScope.
Edit DATA_PATH to point at your Reviews.csv before running.
"""

from pathlib import Path

# ── Dataset ────────────────────────────────────────────────────────────────────
# Place your Reviews.csv inside the data/ folder, or set an absolute path.
DATA_PATH: Path = Path("data/Reviews.csv")

# Number of rows to sample for training (adjust to your machine's RAM/speed).
SAMPLE_SIZE: int = 50_000

# Reproducibility seed
RANDOM_STATE: int = 42
