from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

OUT_DIR = Path("./out")
OUT_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-paper")
matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["savefig.bbox"] = "tight"

RETRIEVAL_COLS = ["relevance", "helpfulness"]
LLM_COLS = RETRIEVAL_COLS + ["correctness"]

# Possible options for each category
HELPFULNESS_OPTIONS = [2, 1, 0, -1, -2]
CORRECTNESS_OPTIONS = [3, 2, 1, 0, -1, -2, -3]
RELEVANCE_OPTIONS = [2, 1, 0, -1, -2]
RATINGS_OPTIONS = {
    "relevance": RELEVANCE_OPTIONS,
    "correctness": CORRECTNESS_OPTIONS,
    "helpfulness": HELPFULNESS_OPTIONS,
}
RATINGS_RANGES = {
    name: (min(vals), max(vals)) for name, vals in RATINGS_OPTIONS.items()
}
