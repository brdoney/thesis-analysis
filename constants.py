from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

OUT_DIR = Path("./out")
OUT_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-paper")
matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["savefig.bbox"] = "tight"
