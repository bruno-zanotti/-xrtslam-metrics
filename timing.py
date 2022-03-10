import numpy as np
from collections import namedtuple
from argparse import ArgumentParser
from pathlib import Path

NS_TO = {"ns": 1, "ms": 1e6, "s": 1e9}

Stats = namedtuple("Stats", ["mean", "std", "min", "q1", "q2", "q3", "max"])


def check_monotonic_rows(rows: np.ndarray) -> None:
    last_ts = 0
    for row in rows:
        ts = row[0]
        assert last_ts < ts
        last_ts = ts


def stats_from_csv(csv_fn: str, i=0, j=-1, divide_by=1) -> Stats:
    rows = np.genfromtxt(csv_fn, delimiter=",", comments="#", dtype=np.int64)
    check_monotonic_rows(rows)
    diffs = (rows[:, j] - rows[:, i]) / divide_by
    return Stats(
        np.mean(diffs),
        np.std(diffs),
        np.min(diffs),
        np.quantile(diffs, 0.25),
        np.median(diffs),
        np.quantile(diffs, 0.75),
        np.max(diffs),
    )


def parse_args():
    parser = ArgumentParser(
        description="Evaluate timing data for Monado visual-inertial tracking",
    )
    parser.add_argument(
        "timing_csv",
        type=Path,
        help="Timing file generated from Monado",
    )
    parser.add_argument(
        "--units",
        type=str,
        help="Time units to show things on",
        default="ms",
        choices=["ns", "ms", "s"],
    )
    return parser.parse_args()


def main():
    user_args = parse_args()
    csv_file = user_args.timing_csv
    units = user_args.units
    s = stats_from_csv(csv_file, divide_by=NS_TO[units])
    print(s)


if __name__ == "__main__":
    main()
