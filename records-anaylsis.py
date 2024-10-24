import json
import re
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from constants import OUT_DIR

chunk_title_pat = re.compile(r".+% ([^\s]+)(?: - page (\d+))?")

chunks: dict[str, int] = {}
new_chunk_id = 0

suffixes: set[str] = set()


class ChunkInfo(NamedTuple):
    post_id: int
    chunk_id: int
    document: str
    doc_type: str
    doc_class: str
    score: float
    page_number: int | None

    @staticmethod
    def from_chunk(
        post_id: int, title: str, chunk: str, score: float, link: str
    ) -> "ChunkInfo":
        global chunks, new_chunk_id
        if chunk in chunks:
            chunk_id = chunks[chunk]
        else:
            chunk_id = new_chunk_id
            chunks[chunk] = chunk_id
            new_chunk_id += 1

        m = chunk_title_pat.match(title)

        if m is None:
            raise ValueError(f"Unable to extract info from title '{title}'")

        # {'', '.rep', '.html', '.txt', '.c', '.s', '.h', '.pptx', '.sh', '.py', '.java', '.md', '.pdf'}
        document, page_number_str = m.groups(default=None)
        if document is None or (page_number_str is None and ".pdf" in title):
            raise ValueError(f"Unable to extract info from title '{title}'")

        page_number = int(page_number_str) if page_number_str else None

        document_file = Path(document)
        suffix = document_file.suffix
        if suffix == "":
            suffix = ".html"
        suffix = suffix.removeprefix(".")

        projects = [
            "/projects/project1",
            "/projects/project2",
            "/projects/project3",
            "/projects/project4",
            "/documents/fuzz",
            "/projects/helpsessions/cush",
            "/projects/helpsessions/threadpool",
            "/projects/helpsessions/malloclab",
            "/projects/cush-handout.pdf",
            "/projects/threadpool-handout.pdf",
            "/projects/malloclab-cs3214.pdf",
            "/projects/pserver-handout.pdf",
        ]
        unit_tests = ["server_unit_test_pserv.py", "server_bench.py"]

        test_parts = ["Test1", "Test_1", "Test_2", "Midterm", "Final"]
        if any(part in document for part in test_parts):
            suffix_type = "Exams"
        elif link.startswith("https://git.cs.vt.edu/cs3214-staff/cs3214-videos"):
            suffix_type = "Lecture Code Examples"
        elif link.startswith("https://git.cs.vt.edu/cs3214-staff"):
            if document in unit_tests:
                suffix_type = "Project Unit Tests"
            else:
                suffix_type = "Project Code"
        elif any(part in link for part in projects):
            suffix_type = "Project Documentation"
        elif suffix in ["html", "md", "pptx", "pdf"]:
            suffix_type = "Lecture Material"
        else:
            raise ValueError(f"Invalid resource {document}")

        suffixes.add(document_file.suffix)

        return ChunkInfo(
            post_id, chunk_id, document, suffix, suffix_type, score, page_number
        )


rows: list[ChunkInfo] = []

records_dir = Path("~/Desktop/records/").expanduser()
for child in records_dir.iterdir():
    if not child.is_file():
        continue

    # doc_link = remove_fragment()
    with child.open() as f:
        child_json: dict[str, Any] = json.load(f)

    post_id: int = child_json["post_id"]
    embed: dict[str, Any]
    for embed in child_json["embeds"]:
        title: str = embed["title"]
        chunk: str = embed["content"]
        link: str = embed["dest"]
        score = float(embed["score"])  # pyright: ignore[reportAny]
        info = ChunkInfo.from_chunk(post_id, title, chunk, score, link)
        rows.append(info)

print(suffixes)
df = pd.DataFrame(rows)
print(df)

chunk_lookup: dict[int, str] = {chunk_id: chunk for chunk, chunk_id in chunks.items()}

# Counts for each class - how distributed are they?
class_counts = df.groupby("doc_class").size()
print(class_counts)
type_counts = df.groupby("doc_type").size()
print(type_counts)
chunk_counts = df.groupby("chunk_id").size().sort_values()
doc_counts = df.groupby("document").size().sort_values()
document_scores = df[["document", "score"]].groupby("document").mean()

# Show the top 10 chunks
# Turns out the tip top are rules from final PDFs, then they're real chunks from project 4 stuff
for i in range(1, 11):
    print(f"==== Top {i} Chunk ====\n", chunk_lookup[chunk_counts.index[-i]])

# Show the top 10 documents
for i in range(1, 11):
    print(f"Top {i} Document:", doc_counts.index[-i], doc_counts.iloc[-i])

# Counts for each chunk - are there spikes? -> bar chart
_ = chunk_counts.plot(
    kind="bar",
    title="Recommended Chunks",
    xlabel="Chunk",
    ylabel="Times Recommended",
    xticks=[],
    width=1,
)
plt.savefig(OUT_DIR / "Resource Chunks.png")
plt.close()

# Counts for each document - are there spikes? -> bar chart
_ = doc_counts.plot(
    kind="bar",
    title="Recommended Documents",
    xlabel="Document",
    ylabel="Times Recommended",
    xticks=[],
    width=1,
)
plt.savefig(OUT_DIR / "Resource Documents.png")
plt.close()

# Counts for each type - how distributed are they?
_ = type_counts.plot(
    kind="bar",
    title="Recommended Resource Types",
    xlabel="Resource Type",
    ylabel="Times Recommended",
)
plt.savefig(OUT_DIR / "Resource Types.png")
plt.close()
# So it's mostly recommending PDFs (assignment specs + slides + help sessions),
# then Python files (test drivers -> students asking about why they're failing tests)
# then webpages (FAQs and instructions for ex5)
# then C files (p4 assignment code)

class_counts.plot(kind="pie", title="Recommended Resource Categories", autopct="%.2f%%")
plt.savefig(OUT_DIR / "Resource Categories.png")
plt.close()
# So code and documentation are about even, all in all, and there's a pretty even spread overall

# Distribution for score -> violin plot or box/whisker

print("Average chunk similarity:", df["score"].mean())
print("Average doc similarity:", document_scores["score"].mean())

_ = sns.violinplot(df, y="score", linewidth=1, bw_adjust=0.5)
plt.title("Chunk Cosine Similarity")
plt.ylabel("Cosine Similarity")
plt.xticks([])
plt.savefig(OUT_DIR / "Chunk Cosine Similarity.png")
plt.close()

_ = sns.violinplot(document_scores, y="score", linewidth=1, bw_adjust=0.5)
plt.title("Document Cosine Similarity")
plt.ylabel("Cosine Similarity")
plt.xticks([])
plt.savefig(OUT_DIR / "Document Cosine Similarity.png")
plt.close()

chunk_df = df["score"].to_frame("score")
chunk_df["type"] = "Chunk"
document_df = document_scores
document_df["type"] = "Document"
scores = pd.concat([chunk_df, document_df])

ax = sns.violinplot(scores, x="type", y="score", linewidth=1, bw_adjust=0.5)
plt.setp(ax.collections, alpha=0.75)
plt.title("Cosine Similarity")
plt.ylabel("Similarity")
plt.xlabel("")
plt.savefig(OUT_DIR / "Cosine Similarity.png")
plt.close()

# Type of resource vs ratings
# Type of resource vs CTR
