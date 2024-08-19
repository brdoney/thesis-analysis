import sqlite3
from typing import Any  # type: ignore[reportAny]
import matplotlib.pyplot as plt
import numpy as np

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
    (res,) = conn.execute(query).fetchone()  # type: ignore[reportAny]
    print(f"{description}: {res}")


print("## Overview")

num_users = query_int("select count(*) from users")
print("Num users:", num_users)

# NOTE: Syntax where clause with use_llm might have to change if sqlite is updated
num_posts = query_int("select count(*) from posts")
num_llm = query_int("select COUNT(*) from posts where use_llm='TRUE'")
num_revtrieval = query_int("select COUNT(*) from posts where use_llm='FALSE'")
print("Num posts:", num_posts)

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

clicks_users: list[int] = sorted(
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

if SHOW_GRAPHS:
    plt.title("Number of Link Clicks Per User")
    _ = plt.bar(range(len(clicks)), clicks, tick_label=users)
    plt.ylabel("Number of Link Clicks")
    plt.xlabel("User ID")
    plt.xticks(rotation=90)
    plt.show()


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
