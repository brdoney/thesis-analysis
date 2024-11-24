import json
import re
import sqlite3
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from constants import LLM_COLS, OUT_DIR, RETRIEVAL_COLS
from fits import *

chunk_title_pat = re.compile(r".+% ([^\s]+)(?: - page (\d+))?")

chunks: dict[str, int] = {}
new_chunk_id = 0

suffixes: set[str] = set()


def get_suffix(document_name: str) -> str:
    document = Path(document_name)
    suffix = document.suffix
    if suffix == "":
        suffix = ".html"
    return suffix.removeprefix(".")


def get_suffix_type(document_name: str, link: str) -> str:
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
    if any(part in document_name for part in test_parts):
        return "Exams"
    elif link.startswith("https://git.cs.vt.edu/cs3214-staff/cs3214-videos"):
        return "Lecture Code Examples"
    elif link.startswith("https://git.cs.vt.edu/cs3214-staff"):
        if document_name in unit_tests:
            return "Project Unit Tests"
        else:
            return "Project Code"
    elif any(part in link for part in projects):
        return "Project Documentation"
    elif get_suffix(document_name) in ["html", "md", "pptx", "pdf"]:
        return "Lecture Material"
    else:
        raise ValueError(f"Invalid resource {document_name}")


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

        document, page_number_str = m.groups(default=None)
        if document is None or (page_number_str is None and ".pdf" in title):
            raise ValueError(f"Unable to extract info from title '{title}'")

        page_number = int(page_number_str) if page_number_str else None

        suffix = get_suffix(document)
        suffixes.add(suffix)
        suffix_type = get_suffix_type(document, link)

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
print(df[["chunk_id", "document"]])
document_lookup: dict[int, str] = {
    chunk_id: document
    for chunk_id, document in df[["chunk_id", "document"]].itertuples(index=False)
}

# Counts for each class - how distributed are they?
class_counts = df.groupby("doc_class").size()
print(class_counts)
type_counts = df.groupby("doc_type").size().sort_values(ascending=False)
print(type_counts)
chunk_counts = df.groupby("chunk_id").size().sort_values(ascending=False)
doc_counts = df.groupby("document").size().sort_values(ascending=False)
document_scores = df[["document", "score"]].groupby("document").mean()

# PDFs are by far the highest, so here:
# type_doc_counts = (
#     df.groupby(["document", "doc_type"])
#     .size()
#     .sort_values(ascending=False)
#     .to_frame("size")
#     .reset_index()
# )
# pdfs = type_doc_counts[type_doc_counts["doc_type"] == "pdf"]
# print("Summary", pdfs, len(pdfs), pdfs["size"].sum())
# Print documents of types with low recommendations
# print(type_doc_counts[type_doc_counts["doc_type"].isin(["s", "rep", "txt", "java", "pptx"])])

# Show the top 10 chunks
# Turns out the tip top are rules from final PDFs, then they're real chunks from project 4 stuff
with open(OUT_DIR / "Top Chunks.txt", "w") as f:
    for i in range(0, 10):
        chunk_id = chunk_counts.index[i]
        chunk_recs = chunk_counts.iloc[i]
        chunk_content = chunk_lookup[chunk_id]
        print(
            f"==== Top {i} Chunk | id={chunk_id} #recs={chunk_recs} ====\n",
            chunk_content,
            file=f,
        )

# Show the top 10 documents

with open(OUT_DIR / "Top Documents.txt", "w") as f:
    for i in range(1, 11):
        print(f"Top {i} Document:", doc_counts.index[-i], doc_counts.iloc[-i], file=f)

    f.write("\n")

    top_docs = doc_counts.iloc[:10].reset_index(name="count")
    top_docs["description"] = "placeholder"
    top_docs["document"] = "\\verb|" + top_docs["document"] + "|"
    f.write(top_docs.to_latex(index=False))

# chunk_id vs. times recommended seems logarithmic (relatively straight line on a logx graph)
# chunk_counts.plot(
#     kind="bar",
#     title="Recommended Chunks",
#     xlabel="Chunk",
#     ylabel="Times Recommended",
#     xticks=[],
#     width=1,
#     logx=True,
# )
# plt.show()


# Counts for each chunk - are there spikes? -> bar chart
_ = chunk_counts.plot(
    kind="bar",
    title="Recommended Chunks",
    xlabel="Chunk",
    ylabel="Times Recommended",
    xticks=[],
    width=1,
)
# fit_zipf(chunk_counts)
# fit_zipfian(chunk_counts)
# fit_exponential(chunk_counts)
# fit_logpoly(chunk_counts)
fit_log(chunk_counts)
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
fit_zipf(doc_counts)
# fit_zipfian(chunk_counts)
# fit_exponential(doc_counts)
# fit_logpoly(doc_counts)
plt.savefig(OUT_DIR / "Resource Documents.png")
plt.close()

# Stacked chunk bar chart (takes a LONG time b/c of the number of chunks)
# chunk_counts_df = chunk_counts.reset_index(name="count")
# chunk_counts_df["document"] = chunk_counts_df["chunk_id"].map(document_lookup)
#
# pivot_df = chunk_counts_df.pivot(
#     index="document", columns="chunk_id", values="count"
# ).fillna(0)
#
# # Sort by document type total
# pivot_df = pivot_df.reindex(doc_counts.index)
#
# # Plot the stacked bar chart
# ax = pivot_df.plot(
#     kind="bar",
#     stacked=True,
#     title="Recommended Documents",
#     xlabel="Document",
#     ylabel="Times Recommended",
#     xticks=[],
#     width=1,
#     legend=False,
# )
# fit_zipf(doc_counts)
#
# plt.savefig(OUT_DIR / "Resource Documents Stacked.png")
# plt.close()


# Counts for each type - how distributed are they?
ax = type_counts.plot(
    kind="bar",
    title="Recommended Resource Types",
    xlabel="Resource Type",
    ylabel="Times Recommended",
    rot=0,
)
plt.bar_label(ax.containers[0])
plt.savefig(OUT_DIR / "Resource Types.png")
plt.close()
# So it's mostly recommending PDFs (assignment specs + slides + help sessions),
# then Python files (test drivers -> students asking about why they're failing tests)
# then webpages (FAQs and instructions for ex5)
# then C files (p4 assignment code)

# Stacked version of types graph (takes a while)
# doc_counts_df = doc_counts.reset_index(name="count")
# doc_counts_df["doc_type"] = doc_counts_df["document"].apply(get_suffix)
#
# pivot_df = doc_counts_df.pivot(
#     index="doc_type", columns="document", values="count"
# ).fillna(0)
#
# # Sort by document type total
# pivot_df = pivot_df.reindex(type_counts.index)
#
# # Plot the stacked bar chart
# ax = pivot_df.plot(
#     kind="bar",
#     stacked=True,
#     title="Recommended Resource Types",
#     xlabel="Resource Type",
#     ylabel="Times Recommended",
#     rot=0,
#     legend=False,
# )
#
# # Bar labels (can't use ax.bar_label b/c it's stacked with an uneven number per bar)
# for idx, total in enumerate(type_counts):
#     ax.text(
#         idx,
#         total,  # Position above the bar
#         str(total),  # Text is the total count
#         ha="center",
#         va="bottom",
#         fontsize=10,
#         color="black",
#     )
#
# plt.savefig(OUT_DIR / "Resource Types Stacked.png")
# plt.close()

# Pie chart version of the resource type bar chart
ax = type_counts.plot(
    kind="pie", labels=None, title="Recommended Resource Types", radius=1.2
)
labels = [f"{doc_type} - {num}" for doc_type, num in type_counts.items()]
# Honestly idek why but this puts it to the left of the pie chart
plt.legend(ax.patches, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig(OUT_DIR / "Resource Types Pie.png")
plt.close()

class_counts.plot(kind="pie", title="Recommended Resource Categories", autopct="%.2f%%")
plt.savefig(OUT_DIR / "Resource Categories.png")
plt.close()
# So code and documentation are about even, all in all, and there's a pretty even spread overall

# Similarity overview stats info
print("Average chunk similarity:", df["score"].mean())
print("Average doc similarity:", document_scores["score"].mean())

# Distribution for score -> violin plot or box/whisker
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

document_scores_dict = dict(zip(document_scores.index, document_scores["score"]))
df["above_doc"] = df[["document", "score"]].apply(
    lambda row: document_scores_dict[row["document"]] > row["score"], axis=1
)
num_above = df["above_doc"].sum()
num_below = len(df) - num_above
print(f"Chunks above: {num_above/len(df):.2%} ({num_above})")
print(f"Chunks below: {num_below/len(df):.2%} ({num_below})")

# How many queries were the Python driver files included in?
num_posts = df["post_id"].nunique()
num_posts_per_doc = df.groupby("document")["post_id"].nunique()
num_unit_test_posts = num_posts_per_doc.loc["server_unit_test_pserv.py"]
print(
    f"Posts with unit tests: {(num_unit_test_posts / num_posts):.2%} ({num_unit_test_posts})"
)
avg_post_per_file = num_posts_per_doc.mean()
print(
    f"Average posts per file: {(avg_post_per_file / num_posts):.2%} ({avg_post_per_file})"
)

# What was the average number of times it was recommended per post?
doc_counts_per_post = (
    df.groupby(["post_id", "document"]).size().reset_index(name="count")
)
grouped = doc_counts_per_post.groupby("document")["count"].sum() / num_posts
sorted_doc_counts = grouped.sort_values(ascending=False)
print("Repetitions per post:")
print(sorted_doc_counts)

# Average number of times it was included in posts where it was included at least once?
average_counts = (
    doc_counts_per_post.groupby("document")["count"].mean().sort_values(ascending=False)
)
print(
    "Number links to unit tests by posts with unit tests at least once:",
    average_counts.loc["server_unit_test_pserv.py"],
)


# How many times does a post repeat a document on average?
reps_per_post = doc_counts_per_post[["post_id", "count"]].groupby("post_id").max()
avg_reps_per_post = reps_per_post["count"].mean()
std_reps_per_post = reps_per_post["count"].std()
med_reps_per_post = reps_per_post["count"].median()
print(
    f"Document repetitions per post: μ={avg_reps_per_post} σ={std_reps_per_post}, median={med_reps_per_post}"
)

# How many documents are there per post on average?
docs_per_post = (
    doc_counts_per_post.groupby("post_id").size().reset_index(name="num_docs")
)
avg_docs_per_post = docs_per_post["num_docs"].mean()
print(f"Average num document per post: {avg_docs_per_post}")

# Type of resource vs ratings
conn = sqlite3.connect("./linkdata.db")

# Minimum number of rows to be included in the graph
ROW_MIN_THRESHOLD = 10

for table, rating_cols, title in [
    ("retrieval_reviews", RETRIEVAL_COLS, "Retrieval"),
    ("llm_reviews", LLM_COLS, "LLM"),
]:
    ret_df = pd.read_sql(f"select * from {table}", conn)
    rows = []
    for doc_type in df["doc_type"].unique():
        posts_with_doc_type = df[df["doc_type"] == doc_type]["post_id"].unique()

        # Skip types with fewer rows than the threshold
        if len(posts_with_doc_type) < ROW_MIN_THRESHOLD:
            continue

        reviews_with_doc_type = ret_df["post_id"].isin(posts_with_doc_type)
        avgs = ret_df[reviews_with_doc_type][rating_cols].mean()
        avgs["doc_type"] = doc_type
        rows.append(avgs)
        # print(f"==={doc_type}===")
        # print(avgs)

    doc_type_avg_df = pd.DataFrame(rows)
    doc_type_avg_df.dropna(inplace=True)
    print(doc_type_avg_df)

    ax = doc_type_avg_df.set_index("doc_type")[rating_cols].plot(kind="bar")

    # Draw reference line at y=0 (not necessary after filtering doc types based on ROW_MIN_THRESHOLD)
    # ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")

    plt.xlabel("Document Type")
    plt.ylabel("Metric Average")
    plt.title(f"{title} Metrics Averages for Each Document Type")
    plt.xticks(rotation=45)
    plt.legend(title="Metric")
    plt.savefig(OUT_DIR / f"{title} Filetype Average.png")
    plt.close()

# Type of resource vs CTR
