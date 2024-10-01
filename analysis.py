from collections.abc import Iterable
import sqlite3
from typing import Any  # type: ignore[reportAny]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from bucketcounter import BucketCounter

plt.style.use("seaborn-v0_8-paper")

conn = sqlite3.connect("./linkdata.db")
# SHOW_GRAPHS = True
SHOW_GRAPHS = False

RETRIEVAL_COLS = ["relevance", "helpfulness"]
LLM_COLS = RETRIEVAL_COLS + ["correctness"]


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
num_llm = query_int("select COUNT(*) from posts where use_llm='TRUE'")
num_revtrieval = query_int("select COUNT(*) from posts where use_llm='FALSE'")
print("Num posts:", num_posts)

num_llm_reviews = query_int("select count(*) from llm_reviews")
num_retrieval_reviews = query_int("select count(*) from retrieval_reviews")
num_reviews = num_llm_reviews + num_retrieval_reviews
print(f"Num reviews: f{num_reviews}")

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

if SHOW_GRAPHS:
    _ = plt.title("Number of Link Clicks Per User")
    _ = plt.bar(range(len(clicks)), clicks, tick_label=users)
    _ = plt.ylabel("Number of Link Clicks")
    _ = plt.xlabel("User ID")
    _ = plt.xticks(rotation=90)
    _ = plt.show()


print("### Per Post")
print("Average link clicks per post (overall CTR):", num_clicks / num_posts)
# If we think of this as a click-through rate, it's better than ads, but still
# not good for an educational tool (I would have expected ~50%)
query_print(
    "Average link clicks per post with at least one click",
    "select avg(click_count) from (select count(*) as click_count from clicks group by post_id)",
)
# This isn't that much higher than 1, suggesting that users aren't clicking
# multiple links often at all, even if they've clicked one. So either our link
# relevance or previews are good, or link relevance is trash.
llm_ctr = llm_clicks / num_llm
retrieval_ctr = retrieval_clicks / num_revtrieval
print("Average clicks per LLM post (LLM CTR):", llm_ctr)
# So LLM CTR is bringing down the average CTR, not that it matters much given
# how low it is
print("Average clicks per Retrieval post (retrieval CTR):", retrieval_ctr)
# Number of clicks is similarly low for each post, with a slight edge going to
# retrieval posts (which makes sense, given that's all they do). Still, the
# difference is small enough to reflect a disinterest in links/satisfaction
# with document previews. The low rate of link clicks per retrieval post
# furthers this, given that the only value it provides is showing relevant
# snippets if users do not clicks the links themselves.

retrieval_relative_ctr = retrieval_ctr / llm_ctr
print(f"Retrieval CTR is {retrieval_relative_ctr:.3f}x better than LLM CTR")
# Unfortunately, there's no way to track how many students actually used (just reviewed)
# with the LLM response, so it is possible that many of them generated it
# simply to have it. As in, it's better to have it than not, even if it's bad.

print("\n## Posts")

print("Num posts:", num_posts)
print("├ LLM:", num_llm)
print("└ Retrieval:", num_revtrieval)
print(f"There are {num_llm/num_revtrieval:.3f}x more LLM posts than retrieval posts")
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
print(f"├ LLM: {num_llm_reviews} ({num_llm_reviews / num_reviews:.2%})")
print(
    f"└ Retreival: {num_retrieval_reviews} ({num_retrieval_reviews / num_reviews:.2%})"
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
# TODO: Check if the author is generally the sole review
# TODO: Check the average number of non-author reviews

# Average scores for LLM
query_print_averages(LLM_COLS, "llm_reviews")
# Average scores for retrieval
query_print_averages(RETRIEVAL_COLS, "retrieval_reviews")


def graph_review_count(
    review_counts: Iterable[int],
    title: str,
    buckets: list[tuple[int, int]] | None = None,
) -> None:
    counter = BucketCounter(review_counts, buckets)

    num_counted_users = counter.total()
    # counter["0"] = num_users - num_counted_users

    labels, counts = counter.sorted_items()

    def autopct_func(percent: float):
        count = round(percent / 100 * num_counted_users)
        return f"{percent:1.2f}% ({count})"

    _ = plt.pie(
        counts,
        labels=labels,
        autopct=autopct_func,
        pctdistance=0.75,
        wedgeprops=dict(linewidth=1, edgecolor="w"),
    )
    _ = plt.title(title)
    _ = plt.show()


# TODO: Scatter plot of number of questions vs. number of reviews
# TODO: Scatter plot of number of questions vs. helpfulness, relevance, etc.
#       SELECT * from "llm_reviews" GROUP BY post_id having count(*) > 1 LIMIT 200;

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
    (num_retrieval_reviews_by_user, "retrieval", [(6, 8)]),
    (num_llm_reviews_by_user, "llm", [(10, 12)]),
    (num_total_reviews_by_user, "total", [(1, 2), (12, 15)]),
]:
    print(f"### {name.title()} review info")
    # Number of unique authors who left reviews
    print(f"- Num authors:", len(data))

    # Reviews per author
    print(f"- Average per user:", np.average(data))
    print(f"- Stddev per user:", np.std(data))

    if SHOW_GRAPHS:
        graph_review_count(
            data,
            f"Number of {name} reviews per user",
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
    ("llm_reviews", llm_df, LLM_COLS),
    ("retrieval_reviews", retrieval_df, RETRIEVAL_COLS),
]:
    averages = []
    for i in range(len(df) - WINDOW_LEN + 1):
        rows = df[i : i + WINDOW_LEN][cols]
        averages.append(rows.mean())
    window_averages = pd.DataFrame(averages)

    if SHOW_GRAPHS:
        ax = window_averages.plot(
            title=f"Rolling Average of {WINDOW_LEN} Reviews",
            ylabel="Average Value",
            xlabel="Window Starting Point (Review Index)",
            xticks=range(
                0, len(window_averages), WINDOW_LEN
            ),  # Set xticks every WIDNOW_LEN points
        )

    print(f"T-test and p-values for {name} rolling averages:")
    for i, col in enumerate(cols):
        overall_average = df[col].mean()
        t_stat, p_value = stats.ttest_1samp(window_averages[col], overall_average)
        print(
            f"- {col}: proposed mean: {window_averages[col].mean()} overall mean: {overall_average} -> p-value {p_value}"
        )

        alternative = "greater"
        # This combo is the only negative slope
        if name == "retrieval_reviews" and col == "relevance":
            alternative = "less"

        linreg = stats.linregress(
            window_averages.index, window_averages[col], alternative=alternative
        )
        print("  ", linreg)

        if SHOW_GRAPHS:
            color = ax.get_lines()[i].get_color()
            y = slope * window_averages.index + intercept
            ax.plot(window_averages.index, y, color=color, linestyle="dashed")

    if SHOW_GRAPHS:
        plt.show()

# LLM
# - All have an extremely small p-value, so slope is definitely positive, meaning that students were getting better over time.
# - The slope of correctness > helpefulness > relevance
# - The actual averages are correctness > relevance > helpfulness though
# Retrieval
# - Again, all have small p-values, so slopes are definitely correct
# - Slope for relevance is negative, slope for helpfulness is positive
# - Helpfulness started low, but was matching than relevance by the end

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

print("### LLM total time info")
for col in LLM_COLS:
    x = llm_gen_df["total_time"]
    linreg = stats.linregress(x, llm_gen_df[col])
    spearman = stats.spearmanr(x, llm_gen_df[col])

    print("-", col, linreg, spearman)

    if SHOW_GRAPHS:
        llm_gen_df.plot(
            x="total_time",
            y=col,
            kind="scatter",
            title=f"Total Retrieval and Generation Time vs. {col.title()}",
            xlabel="Total Time",
            ylabel=col.title(),
        )
        # NOTE: These lines have no statistical significance and should probably be removed
        plt.plot(x, linreg.slope * x + linreg.intercept, linestyle="dashed")
        plt.show()

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

print("### Retrieval time info")
for col in RETRIEVAL_COLS:
    x = retrieval_gen_df["retrieval_time"]
    linreg = stats.linregress(x, retrieval_gen_df[col])
    spearman = stats.spearmanr(x, retrieval_gen_df[col])

    print("-", col, linreg, spearman)

    if SHOW_GRAPHS:
        retrieval_gen_df.plot(
            x="retrieval_time",
            y=col,
            kind="scatter",
            title=f"Retrieval Time vs. {col.title()}",
            xlabel="Retrieval Time",
            ylabel=col.title(),
        )
        # NOTE: These lines have almost no statistical significance and should probably be removed
        plt.plot(x, linreg.slope * x + linreg.intercept, linestyle="dashed")
        plt.show()

# So the closest to being statistically significant is helpfulness, which had a
# p-value of 0.066. But ultimately, none of the lines here are statistically
# significant, which makes sense for the same reasons as for the LLMs.
