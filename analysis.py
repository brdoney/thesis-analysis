from collections.abc import Iterable
import sqlite3
from typing import Any # type: ignore[reportAny]

import matplotlib.pyplot as plt
import numpy as np

from bucketcounter import BucketCounter

plt.style.use("seaborn-v0_8-paper")

conn = sqlite3.connect("./linkdata.db")
SHOW_GRAPHS = True


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
    print(f"Average values for {table}")
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
print(f"└ Retreival: {num_retrieval_reviews} ({num_retrieval_reviews / num_reviews:.2%})")
retrieval_cols = ["relevance", "helpfulness"]
llm_cols = retrieval_cols + ["correctness"]
# Average scores for LLM
query_print_averages(llm_cols, "llm_reviews")
# Average scores for retrieval
query_print_averages(retrieval_cols, "retrieval_reviews")


def graph_review_count(
    review_counts: Iterable[int], title: str, buckets: list[tuple[int, int]] | None = None
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
# TODO: Number of reviews on the same post - how distributed are the reviews?
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
retrieval_reviews: tuple[int, ...]
llm_reviews: tuple[int, ...]
total_reviews: tuple[int, ...]
[retrieval_reviews, llm_reviews, total_reviews] = zip(*query_list(query))

for data, name, bucket in [
    (retrieval_reviews, "retrieval", [(6, 8)]),
    (llm_reviews, "llm", [(10, 12)]),
    (total_reviews, "total", [(1, 2), (12,15)]),
]:
    # Number of unique authors who left reviews
    print(f"Num {name} review authors:", len(data))

    # Reviews per author
    print(f"Average {name} reviews per user:", np.average(data))
    print(f"Stddev {name} retrieval reviews per user:", np.std(data))

    if SHOW_GRAPHS:
        graph_review_count(
            data,
            f"Number of {name} reviews per user",
            bucket,
        )

num_5plus_users = sum(1 for review in total_reviews if review >= 5)
print(
    f"Number of users with 5 or more reviews: {num_5plus_users} ({num_5plus_users / num_users:.2%})"
)

# What were the other users who have no reviews doing? - They were making posts but not reviewing them
