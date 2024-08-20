from collections import Counter
from collections.abc import Iterable
from functools import cmp_to_key


class BucketCounter(Counter[str]):
    def __init__(
        self, iterable: Iterable[int], buckets: list[tuple[int, int]] | None = None
    ) -> None:
        super().__init__()

        if buckets is None:
            self.update([str(k) for k in iterable])
            return

        key_dict: dict[int, str] = {}
        for start, end in buckets:
            key = f"{start}-{end}"
            for i in range(start, end + 1):
                key_dict[i] = key

        for d in iterable:
            if d in key_dict:
                key = key_dict[d]
            else:
                key = str(d)
            self[key] = self.get(key, 0) + 1

    def sorted_items(self) -> tuple[list[str], list[int]]:
        def key_func(item: tuple[str, int]) -> tuple[int, str]:
            key = item[0]
            if (i := key.find("-")) != -1:
                key = key[:i]
            return (len(key), key)

        labels: list[str]
        counts: list[int]
        labels, counts = zip(*sorted(self.items(), key=key_func))
        return labels, counts
