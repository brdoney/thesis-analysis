from os import PathLike
from constants import OUT_DIR, SURVEY_DIR

import pandas as pd
import matplotlib.pyplot as plt

OTHER_TOOLS_DIR = SURVEY_DIR / "other tools"


def print_percents(csv_path: PathLike, num_students: int) -> None:
    df = pd.read_csv(csv_path)
    df["Percent Students"] = df["Number"] / num_students
    df.sort_values(by="Percent Students", inplace=True, ascending=False)
    print(df)
    print()


# Our tool
print_percents(SURVEY_DIR / "uses.csv", 32)

# Other tools
print_percents(OTHER_TOOLS_DIR / "uses.csv", 20)
print_percents(OTHER_TOOLS_DIR / "familiarity.csv", 20)

with open(OUT_DIR / "Open Response Table.txt", "w") as f:
    f.write(pd.read_csv(SURVEY_DIR / "open_response.csv").to_latex(index=False))
