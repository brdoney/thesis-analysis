import sqlite3
from collections.abc import Iterable
from typing import Any  # type: ignore[reportAny]

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from bucketcounter import BucketCounter
from constants import OUT_DIR

conn = sqlite3.connect("./linkdata.db")
# SHOW_GRAPHS = True
SHOW_GRAPHS = False

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


def query_list(query: str) -> list[Any]:
    return conn.execute(query).fetchall()


def query_int(query: str) -> int:
    res: int
    (res,) = conn.execute(query).fetchone()
    return res


def query_float(query: str) -> float:
    res: float
    (res,) = conn.execute(query).fetchone()
    return res


def query_print(description: str, query: str) -> None:
    (res,) = conn.execute(query).fetchone()  # pyright: ignore[reportAny]
    print(f"{description}: {res}")


def query_print_averages(cols: list[str], table: str) -> None:
    avg_str = ", ".join(f"avg({col})" for col in cols)
    query = f"select {avg_str} from {table}"
    res: tuple[float, ...]
    res = conn.execute(query).fetchone()
    print(f"### Average values for {table}")
    for col, val in zip(cols, res):
        print(f"- {col.title()}: {val}")


def pie_chart(
    data: Iterable[int],
    title: str,
    buckets: list[tuple[int, int]] | None = None,
    data_range: tuple[int, int] | None = None,
) -> None:
    counter = BucketCounter(data, buckets)

    num_counted_users = counter.total()
    # counter["0"] = num_users - num_counted_users

    labels, counts = counter.sorted_items()

    if data_range is not None:
        title += f" ({data_range[0]} to {data_range[1]})"

    def autopct_func(percent: float):
        count = round(percent / 100 * num_counted_users)
        return f"{percent:1.2f}% ({count})"

    _ = plt.pie(
        counts,
        labels=labels,
        autopct=autopct_func,
        pctdistance=0.75,
        wedgeprops=dict(linewidth=1, edgecolor="w"),
        textprops=dict(fontsize=6),
    )

    _ = plt.title(title)
    plt.savefig(OUT_DIR / f"{title}.png")

    if SHOW_GRAPHS:
        plt.show()

    plt.close()


def violin_plot(df: pd.DataFrame, title: str) -> None:
    # bw_adjust to decrease smoothing (since data is discrete)
    ax = sns.violinplot(
        data=df, bw_adjust=0.5, linewidth=1, formatter=lambda s: s.title()
    )
    # Make background colors less harsh
    plt.setp(ax.collections, alpha=0.75)

    _ = plt.title(title)

    plt.savefig(OUT_DIR / f"{title}.png")

    if SHOW_GRAPHS:
        plt.show()

    plt.close()


def graph_reviews(cols: list[str], table: str, title: str) -> None:
    query = f"select {','.join(cols)} from {table}"
    df = pd.read_sql(query, conn)

    for col in cols:
        pie_chart(df[col], f"{title} {col.title()}", [], RATINGS_RANGES[col])

    violin_plot(df, f"{title} Ratings")


def flatten[T](items: list[tuple[T]]) -> list[T]:
    if len(items[0]) == 1:
        return [item[0] for item in items]
    else:
        return [it for item in items for it in item]


print("## Overview")

# NOTE: Syntax where clause with use_llm might have to change if sqlite is updated
# Only have to check consent for this table, since no other tables get data for user if consent is false
num_users = query_int("select count(*) from users where consent='TRUE'")
print("Num users:", num_users)

num_posts = query_int("select count(*) from posts")
num_llm_posts = query_int("select COUNT(*) from posts where use_llm='TRUE'")
num_retrieval_posts = query_int("select COUNT(*) from posts where use_llm='FALSE'")
print("Num posts:", num_posts)

num_llm_reviews = query_int("select count(*) from llm_reviews")
num_retrieval_reviews = query_int("select count(*) from retrieval_reviews")
num_reviews = num_llm_reviews + num_retrieval_reviews
print(f"Num reviews: {num_reviews}")

num_clicks = query_int("select count(*) from clicks")
print("Total link clicks", num_clicks)

print("\n## Link clicks")
query_print(
    "Number of users who clicked a link", "select count(distinct user_id) from clicks"
)

print("Total link clicks", num_clicks)
llm_clicks = query_float(
    "select count(*) from clicks inner join posts on posts.id=clicks.post_id where use_llm='TRUE'"
)
retrieval_clicks = query_float(
    "select count(*) from clicks inner join posts on posts.id=clicks.post_id where use_llm='FALSE'"
)
deleted_clicks = num_clicks - llm_clicks - retrieval_clicks
print("├ LLM:", llm_clicks)
print("├ Retrieval:", retrieval_clicks)
print("└ Deleted post:", deleted_clicks)
# So most link clicks are on LLM posts (because of the volume)

print("Clicks per user", num_clicks / num_users)
# So users definitely aren't clicking many links
query_print(
    "Clicks per user who clicked at least once",
    "select avg(click_count) from (select count(*) as click_count from clicks group by user_id)",
)
# So if a user clicks one link, they're likely to click multiple

clicks_users: list[tuple[int, int]] = sorted(
    query_list("select count(*), user_id from clicks group by user_id")
)
clicks: tuple[int, ...]
users: tuple[int, ...]
clicks, users = zip(*clicks_users)

clicks_stdev = np.std(clicks)
print(f"Stddev link clicks: {clicks_stdev}")

users_with_clicks = len(users)
users_no_clicks = num_users - users_with_clicks
print(
    f"Users who clicked a link: {users_with_clicks}/{num_users} | Users who didn't: {users_no_clicks}/{num_users} -> {users_with_clicks/users_no_clicks}"
)
# So about a quarter of students are clicking on links

_ = plt.bar(range(len(clicks)), clicks, tick_label=users)
_ = plt.title("Number of Link Clicks Per User")
_ = plt.ylabel("Number of Link Clicks")
_ = plt.xlabel("User ID")
# We don't really care about the exact user ID
_ = plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
# If we did though, this is good:
# _ = plt.xticks(rotation=90, fontsize=7)
plt.savefig(OUT_DIR / "Clicks Per User.png")

if SHOW_GRAPHS:
    plt.show()

plt.close()


print("### Per Post")
print("Average link clicks per post (overall CTR):", num_clicks / (num_posts * 4))
# If we think of this as a click-through rate, it's better than ads, but still
# not good for an educational tool (I would have expected ~50%)
query_print(
    "Average link clicks per post with at least one click",
    "select avg(click_count) from (select count(*) as click_count from clicks group by post_id)",
)
# This isn't that much higher than 1, suggesting that users aren't clicking
# multiple links often at all, even if they've clicked one. So either our link
# relevance or previews are good, or link relevance is trash.
llm_ctr = llm_clicks / (num_llm_posts * 4)
retrieval_ctr = retrieval_clicks / (num_retrieval_posts * 4)
print("Average clicks per LLM post (LLM CTR):", llm_ctr)
# So LLM CTR is bringing down the average CTR, not that it matters much given
# how low it is
print("Average clicks per retrieval-only post (retrieval CTR):", retrieval_ctr)
# Number of clicks is similarly low for each post, with a slight edge going to
# retrieval posts (which makes sense, given that's all they do). Still, the
# difference is small enough to reflect a disinterest in links/satisfaction
# with document previews. The low rate of link clicks per retrieval post
# furthers this, given that the only value it provides is showing relevant
# snippets if users do not clicks the links themselves.

retrieval_relative_ctr = retrieval_ctr / llm_ctr
print(f"Retrieval-only CTR is {retrieval_relative_ctr:.3f}x better than LLM CTR")
# Unfortunately, there's no way to track how many students actually used (just reviewed)
# with the LLM response, so it is possible that many of them generated it
# simply to have it. As in, it's better to have it than not, even if it's bad.

print("\n## Posts")

print("Num posts:", num_posts)
print("├ LLM:", num_llm_posts)
print("└ Retrieval:", num_retrieval_posts)
print(
    f"There are {num_llm_posts/num_retrieval_posts:.3f}x more LLM posts than retrieval posts"
)
# So students VASTLY prefer to see the LLM's answer

query_print(
    "Posts per user who posted at least once",
    "select avg(post_count) from (select count(*) as post_count from posts group by author)",
)

print("Posts per user", num_posts / num_users)

print("\n## Timings")
query_print("Average retrieval time", "select avg(retrieval_time) from posts")
query_print("Average generation time", "select avg(generation_time) from posts")
query_print(
    "Average response time", "select avg(retrieval_time + generation_time) from posts"
)

print("\n## Reviews")
print(f"Num reviews: {num_reviews}")

query = """
SELECT COUNT(*) FROM posts p
JOIN llm_reviews r ON p.id = r.post_id AND p.author = r.author
"""
num_llm_author_reviews = query_int(query)
query = """
SELECT COUNT(*) FROM posts p
JOIN retrieval_reviews r ON p.id = r.post_id AND p.author = r.author
"""
num_retrieval_author_reviews = query_int(query)

num_non_author_llm_reviews = num_llm_reviews - num_llm_author_reviews
num_non_author_retrieval_reviews = num_retrieval_reviews - num_retrieval_author_reviews

print(f"├ LLM: {num_llm_reviews} ({num_llm_reviews / num_reviews:.2%})")
print(
    f"├─ Author: {num_llm_author_reviews} ({num_llm_author_reviews/num_llm_reviews:.2%})"
)
print(
    f"├─ Non-author: {num_non_author_llm_reviews} ({num_non_author_llm_reviews/num_llm_reviews:.2%})"
)
print(
    f"├ Retrieval: {num_retrieval_reviews} ({num_retrieval_reviews / num_reviews:.2%})"
)
print(
    f"├─ Author: {num_retrieval_author_reviews} ({num_retrieval_author_reviews/num_retrieval_reviews:.2%})"
)
print(
    f"└─ Non-author: {num_non_author_retrieval_reviews} ({num_non_author_retrieval_reviews / num_retrieval_reviews:.2%})"
)

query = """
select AVG(review_count) from (
    select COUNT(*) as review_count from llm_reviews group by post_id
)
"""
query_print("Average num reviews per LLM post", query)

query = """
select AVG(review_count) from (
    select COUNT(*) as review_count from retrieval_reviews group by post_id
)
"""
query_print("Average num reviews per retrieval post", query)

# Both types of reviews are pretty close to 1 review per post, which makes sense

query = """
SELECT COUNT(DISTINCT post_id) AS posts_with_reviews
FROM (
    SELECT p.id AS post_id
    FROM posts p
    JOIN llm_reviews lr ON p.id = lr.post_id

    UNION

    SELECT p.id AS post_id
    FROM posts p
    JOIN retrieval_reviews rr ON p.id = rr.post_id
)
"""
num_reviewed_posts = query_int(query)
percent_reviewed_posts = num_reviewed_posts / num_posts
print(
    f"Percent posts with >=1 review: {percent_reviewed_posts:.2%} ({num_reviewed_posts})"
)

for review_type, total_posts in [("retrieval", num_posts), ("llm", num_llm_posts)]:
    query = f"""
    SELECT COUNT(DISTINCT post_id)
    FROM (
        SELECT p.id AS post_id
        FROM posts p
        JOIN {review_type}_reviews r ON p.id = r.post_id
    );
    """
    num_reviewed_posts = query_int(query)
    percent_reviewed_posts = num_reviewed_posts / total_posts
    print(
        f"Percent {review_type} posts with >=1 review: {percent_reviewed_posts:.2%} ({num_reviewed_posts})"
    )

    query = f"""
    SELECT COUNT(*) FROM posts p
    JOIN {review_type}_reviews r ON p.id = r.post_id AND p.author = r.author
    """
    num_author_reviewed = query_int(query)
    percent_author_reviewed = num_author_reviewed / total_posts
    print(
        f"Percent {review_type} posts with an author review: {percent_author_reviewed:.2%} ({num_author_reviewed})"
    )
    query = f"""
    SELECT COUNT(*) from (
        SELECT COUNT(DISTINCT p.id) FROM posts p
        JOIN {review_type}_reviews r ON p.id = r.post_id
        GROUP BY p.id
        HAVING COUNT(*) = 1 AND r.author = p.author
    )
    """

    num_sole_author_reviewed = query_int(query)
    num_non_author_reviewed = num_reviewed_posts - num_sole_author_reviewed
    percent_sole_author_reviewed = num_sole_author_reviewed / total_posts
    print(
        f"Percent {review_type} posts with sole author review: {percent_sole_author_reviewed:.2%} ({num_sole_author_reviewed})"
    )

# Average scores for LLM
query_print_averages(LLM_COLS, "llm_reviews")
graph_reviews(LLM_COLS, "llm_reviews", "LLM")

# Average scores for retrieval
query_print_averages(RETRIEVAL_COLS, "retrieval_reviews")
graph_reviews(RETRIEVAL_COLS, "retrieval_reviews", "Retrieval")


query = """
SELECT 
    -- User info
    -- u.id AS user_id, 
    COUNT(DISTINCT p.id) AS post_count,
    COUNT(DISTINCT lr.rowid) AS llm_review_count,
    COUNT(DISTINCT rr.rowid) AS retrieval_review_count,

    -- Average LLM review stats
    AVG(lr.helpfulness) AS avg_lhelpfulness,
    AVG(lr.relevance) AS avg_lrelevance,
    AVG(lr.correctness) AS avg_lcorrectness,

    -- Average retrieval review stats
    AVG(rr.helpfulness) AS avg_rhelpfulness,
    AVG(rr.relevance) AS avg_rrelevance

FROM users u
LEFT JOIN posts p ON u.id = p.author
LEFT JOIN llm_reviews lr ON u.id = lr.author
LEFT JOIN retrieval_reviews rr ON u.id = rr.author
WHERE u.consent="TRUE"
GROUP BY u.id
ORDER BY post_count ASC
"""
per_user_df = pd.read_sql(query, conn)
per_user_df["total_review_count"] = (
    per_user_df["llm_review_count"] + per_user_df["retrieval_review_count"]
)
# To trim the single outlier, just a random student who liked the tool i guess
# Doesn't make a difference in the result (i.e. p-value) though
# questions_v_reviews_df = questions_v_reviews_df[questions_v_reviews_df["post_count"] > 30]

linreg = stats.linregress(
    per_user_df["post_count"],
    per_user_df["total_review_count"],
    alternative="greater",
)
print("Num posts vs num reviews:", linreg)

per_user_df.plot(
    x="post_count",
    y="total_review_count",
    kind="scatter",
    title="Number of Posts vs. Number of Reviews",
    xlabel="Number of Posts",
    ylabel="Number of Reviews",
)
x = per_user_df["post_count"].sort_values()
plt.plot(x, linreg.slope * x + linreg.intercept, linestyle="dashed")
plt.savefig(OUT_DIR / "Posts vs Reviews.png")

if SHOW_GRAPHS:
    plt.show()

plt.close()

# So there's a statistically significant correlation b/t the number of posts and
# number of reviews a student left, which makes sense: more opportunity to
# review a post if you make a bunch of them, since we don't allow duplicates
# and people preferred to review their own posts over those of others


def prmatrix(df: pd.DataFrame, graph_title: str, out_filename: str):
    pvalues = df.corr(method=lambda x, y: stats.spearmanr(x, y).pvalue) - np.eye(
        len(df.columns)
    )
    rvalues = df.corr(method=lambda x, y: stats.spearmanr(x, y).statistic)

    # accept_pvalues = pvalues.map(lambda x: "*" if 0 < x <= 0.05 else "")
    accept_pvalues = pvalues[(0 < pvalues) & (pvalues < 0.05)]
    accept_rvalues = rvalues[pvalues < 0.05]
    with open(OUT_DIR / out_filename, "w") as f:
        _ = f.write(
            f"P values:\n{pvalues.to_string()}\n\nFiltered P values:\n{accept_pvalues.to_string()}\n\nFiltered R values:\n{accept_rvalues.to_string()}"
        )

    _ = sns.heatmap(pvalues)
    _ = plt.title(f"{graph_title} P-values")
    _ = plt.xticks(rotation=45, ha="right")
    plt.savefig(OUT_DIR / f"{graph_title} P-values.png")

    if SHOW_GRAPHS:
        plt.show()

    plt.close()

    # _ = sns.heatmap(accept_rvalues, cmap="Greens", vmin=0)
    _ = sns.heatmap(accept_rvalues, vmin=0)
    _ = plt.title(f"{graph_title} R-values")
    _ = plt.xticks(rotation=45, ha="right")
    plt.savefig(OUT_DIR / f"{graph_title} R-values.png")

    if SHOW_GRAPHS:
        plt.show()

    plt.close()


# Per-user significance stats
prmatrix(per_user_df, "Per-user Statistics", "per-user-matrix.txt")
# So the number of posts and reviews have no statistically significant
# influence over the ratings that people are giving, as opposed to time, which
# did improve the ratings people were giving.
# Can also extract some info from signficance of averages, but it's better to
# just look at the data directly over the averages, since these represent a
# per-user info (e.g. users who rate one LLM category highly are likely to rate
# the others highly as well)
# -> With the exception of retrieval relevance to LLM helpfulness, there is a
# statistically significant positive correlation between a user's average
# rating between statistics. This makes me think that LLM hallucinations makes
# these things irrelevant.

# Is a post whose retrieval has been rated as relevant have better LLM stats?
query = """
SELECT 
    -- LLM
    lr.helpfulness as `LLM Helpfulness`, 
    lr.relevance as `LLM Relevance`, 
    lr.correctness as `LLM Correctness`, 
    -- Retrieval
    rr.relevance as `Retrieval Relevance`, 
    rr.helpfulness as `Retrieval helpfulness`
FROM llm_reviews lr
JOIN retrieval_reviews rr ON lr.post_id = rr.post_id;
"""
llm_retrieval_reviews = pd.read_sql(query, conn)
prmatrix(llm_retrieval_reviews, "Review Statistics", "llm-and-retrieval-reviews.txt")
# So everything is statistically significantly positively correlated, although
# retrieval relevance to LLM stats aren't *as* signficant as the rest of
# them. They're all a positive correlation though, which makes sense.
# - Isn't that much connection between retrieval stats and LLM stats, just a
# slight positive correlation relative to others
# - Strongest correlation is LLM correctness<->helpfulness, which makes sense
# since incorrect LLM answers are worthless
# - LLM helpfulness<->relevance is also high
# TODO: Talk to Dr. Back about this

query = """
SELECT 
    -- LLM
    lr.helpfulness as `LLM Helpfulness`, 
    lr.relevance as `LLM Relevance`, 
    lr.correctness as `LLM Correctness`
FROM llm_reviews lr WHERE lr.correctness = 0;
"""
zero_reviews = pd.read_sql(query, conn)
print(zero_reviews.mean())
print(zero_reviews.mode())


def series_range(series: "pd.Series[int]") -> npt.NDArray[np.int_]:
    return np.arange(series.min(), series.max() + 1, step=1)

def boxplot(df: pd.DataFrame, x: str, y: str) -> None:
    # meanprops = dict(markerfacecolor="C1")
    fig = sns.catplot(
        df,
        x=x,
        y=y,
        kind="violin",
        bw_adjust=0.5,
        linewidth=1,
        # kind="box",
        # showmeans=True,
        # meanprops=meanprops,
    )
    # _ = sns.pointplot(
    #     df,
    #     x=x,
    #     y=y,
    #     estimator=np.mean,
    #     linestyle="none",
    #     ax=fig.ax,
    #     color="C1",
    # )

    title = f"{x} vs {y}"
    _ = plt.title(title)
    plt.savefig(OUT_DIR / f"{title}.png")

    if SHOW_GRAPHS:
        plt.show()

    plt.close(fig.figure)

    # Bubble version of the same chart
    counted = df.groupby([x, y]).size().reset_index(name="count")
    counted_range: tuple[int, int] = (counted["count"].min(), counted["count"].max())
    scale = 5
    sizes = (scale * counted_range[0], scale * counted_range[1])

    _ = sns.scatterplot(counted, x=x, y=y, size="count", sizes=sizes, legend=False)

    plt.xticks(series_range(counted[x]))
    plt.yticks(series_range(counted[y]))

    _ = plt.title(title)
    plt.savefig(OUT_DIR / f"{title} bubble.png")
    if SHOW_GRAPHS:
        plt.show()
    plt.close()


# Is the LLM covering for us when we get poor retrieval stats?
# Graph average retrieval relevance vs LLM helpfulness and see what spread is like - is it linear? Clustered?
# print(len(llm_retrieval_reviews))
boxplot(llm_retrieval_reviews, "Retrieval Relevance", "LLM Helpfulness")
# boxplot(llm_retrieval_reviews, "LLM Correctness", "LLM Helpfulness")
boxplot(llm_retrieval_reviews, "LLM Relevance", "LLM Helpfulness")


query = """
SELECT 
    COALESCE(r1.review_count, 0) as retrieval_reviews, 
    COALESCE(r2.review_count, 0) as llm_reviews,
    COALESCE(r1.review_count, 0) + COALESCE(r2.review_count, 0) as total_reviews
FROM users u
    LEFT JOIN (
        SELECT author, COUNT(*) AS review_count FROM retrieval_reviews GROUP BY author
    ) r1 ON u.id = r1.author
    LEFT JOIN (
        SELECT author, COUNT(*) AS review_count FROM llm_reviews GROUP BY author
    ) r2 ON u.id = r2.author
WHERE u.consent='TRUE'
GROUP BY u.id;
"""
num_retrieval_reviews_by_user: tuple[int, ...]
num_llm_reviews_by_user: tuple[int, ...]
num_total_reviews_by_user: tuple[int, ...]
[num_retrieval_reviews_by_user, num_llm_reviews_by_user, num_total_reviews_by_user] = (
    zip(*query_list(query))
)

for data, name, bucket in [
    (num_retrieval_reviews_by_user, "Retrieval", [(6, 8)]),
    (num_llm_reviews_by_user, "LLM", [(10, 12)]),
    (num_total_reviews_by_user, "Total", [(1, 2), (12, 15)]),
]:
    print(f"### {name} Review Info")
    # Number of unique authors who left reviews
    num_authors = sum(1 for d in data if d > 0)
    print("- Num authors:", num_authors)

    # Reviews per author
    print("- Average per user:", np.average(data))
    print("- Stddev per user:", np.std(data))

    pie_chart(
        data,
        f"Number of {name} Reviews Per User",
        bucket,
    )

num_5plus_users = sum(1 for review in num_total_reviews_by_user if review >= 5)
print(
    f"Number of users with 5 or more total reviews: {num_5plus_users} ({num_5plus_users / num_users:.2%})"
)

# What were the other users who have no reviews doing? - They were making posts but not reviewing them

print("\n## Rolling Averages")

llm_df = pd.read_sql(  # pyright: ignore[reportUnknownMemberType]
    "select * from llm_reviews", con=conn, index_col=["post_id", "author"]
)
retrieval_df = pd.read_sql(  # pyright: ignore[reportUnknownMemberType]
    "select * from retrieval_reviews", con=conn, index_col=["post_id", "author"]
)

WINDOW_LEN = 100
for name, df, cols in [
    ("LLM", llm_df, LLM_COLS),
    ("Retrieval", retrieval_df, RETRIEVAL_COLS),
]:
    averages = []
    start_i = int(len(df) / 3)
    for i in range(start_i, len(df) - WINDOW_LEN + 1):
        rows = df[i : i + WINDOW_LEN][cols]
        averages.append(rows.mean())
    window_averages = pd.DataFrame(averages)
    window_averages["window_start"] = pd.Series(range(start_i, len(df) - WINDOW_LEN + 1))
    window_averages = window_averages.set_index("window_start")

    ax = window_averages.plot(
        title=f"Rolling Average of {WINDOW_LEN} Reviews",
        ylabel="Average Value",
        xlabel="Window Start Index",
        xticks=range(
            window_averages.index[0], window_averages.index[-1], WINDOW_LEN
        ),  # Set xticks every WIDNOW_LEN points
    )

    print(f"T-test and p-values for {name} rolling averages:")
    r_values = {}
    for i, col in enumerate(cols):
        overall_average = df[col].mean()
        t_stat, p_value = stats.ttest_1samp(window_averages[col], overall_average)
        print(
            f"- {col}: proposed mean: {window_averages[col].mean()} overall mean: {overall_average} -> p-value {p_value}"
        )

        alternative = "greater"
        # This combo is the only negative slope
        if name == "Retrieval" and col == "relevance":
            alternative = "less"

        linreg = stats.linregress(
            window_averages.index, window_averages[col], alternative=alternative
        )
        print("  ", linreg)

        color = ax.get_lines()[i].get_color()
        y = linreg.slope * window_averages.index + linreg.intercept
        ax.plot(window_averages.index, y, color=color, linestyle="dashed")

        r_values[col] = linreg.rvalue

    handles, labels = ax.get_legend_handles_labels()
    new_labels = [f"{col.title()} (r={r_values[column]:.2f})" for column, col in zip(df.columns, cols)]
    _ = ax.legend(handles, new_labels)

    plt.savefig(OUT_DIR / f"Rolling Avg {name}.png")

    if SHOW_GRAPHS:
        plt.show()

    plt.close()

# LLM
# - All have an extremely small p-value, so slope is definitely positive,
# meaning that students were getting better over time.
# - The slope of correctness > helpefulness > relevance
# - The actual averages are correctness > relevance > helpfulness though
# Retrieval
# - Again, all have small p-values, so slopes are definitely correct
# - Slope for relevance is negative, slope for helpfulness is positive
# - Helpfulness started low, but was matching than relevance by the end
# TODO: Talk with Dr. Back about whether this proves students were getting
# better with the tool

print("\n## Time vs Ratings")

query = """
    select l.post_id, l.author, l.relevance, l.helpfulness, l.correctness, p.retrieval_time + p.generation_time as total_time
    from 'llm_reviews' as l
    inner join posts as p
        on l.post_id = p.id
    order by total_time asc
"""
llm_gen_df = pd.read_sql(query, conn, index_col=["post_id", "author"])

# Remove the single outlier (network issues)
llm_gen_df = llm_gen_df[llm_gen_df["total_time"] < 3]

prmatrix(llm_gen_df, "LLM Time vs Ratings", "llm-time-vs-ratings.txt")

print("### LLM total time info")
for col in LLM_COLS:
    x = llm_gen_df["total_time"]
    linreg = stats.linregress(x, llm_gen_df[col])
    spearman = stats.spearmanr(x, llm_gen_df[col])

    print("-", col, linreg, spearman)

    _ = llm_gen_df.plot(
        x="total_time",
        y=col,
        kind="scatter",
        title=f"Combined Retrieval and Generation Time vs. {col.title()}",
        xlabel="Time (s)",
        ylabel=col.title(),
    )
    # NOTE: These lines have no statistical significance and should probably be removed
    _ = plt.plot(x, linreg.slope * x + linreg.intercept, linestyle="dashed")
    _ = plt.yticks(RATINGS_OPTIONS[col])

    plt.savefig(OUT_DIR / f"LLM Time vs {col.title()}.png")

    if SHOW_GRAPHS:
        plt.show()

    plt.close()

# So total time doesn't significantly correlate with any of the ratings, which
# makes sense since longer time doesn't really mean much when we're streaming
# the result (users won't be dissatisfied; helpfulness), it won't change how
# the vector database works (relevance), and we cap our answer at a low number
# of words (500? correctness)

query = """
    select r.post_id, r.author, r.relevance, r.helpfulness, p.retrieval_time from 'retrieval_reviews' as r
    inner join posts as p
        on r.post_id = p.id
    order by p.retrieval_time asc
"""
retrieval_gen_df = pd.read_sql(query, conn, index_col=["post_id", "author"])

prmatrix(retrieval_gen_df, "Retrieval Time vs Ratings", "retrieval-time-vs-ratings.txt")

print("### Retrieval time info")
for col in RETRIEVAL_COLS:
    x = retrieval_gen_df["retrieval_time"]
    linreg = stats.linregress(x, retrieval_gen_df[col])
    spearman = stats.spearmanr(x, retrieval_gen_df[col])

    print("-", col, linreg, spearman)

    _ = retrieval_gen_df.plot(
        x="retrieval_time",
        y=col,
        kind="scatter",
        title=f"Retrieval Time vs. {col.title()}",
        xlabel="Retrieval Time (s)",
        ylabel=col.title(),
    )
    # NOTE: These lines have almost no statistical significance and should probably be removed
    _ = plt.plot(x, linreg.slope * x + linreg.intercept, linestyle="dashed")
    _ = plt.yticks(RATINGS_OPTIONS[col])

    plt.savefig(OUT_DIR / f"Retrieval Time vs {col.title()}.png")

    if SHOW_GRAPHS:
        plt.show()

    plt.close()

# So the closest to being statistically significant is helpfulness, which had a
# p-value of 0.066. But ultimately, none of the lines here are statistically
# significant, which makes sense for the same reasons as for the LLMs.
