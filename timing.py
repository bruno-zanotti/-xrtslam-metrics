import csv
from typing import List
import numpy as np
from collections import namedtuple
from os import listdir, stat
from os.path import exists
from tabulate import tabulate
from argparse import ArgumentParser
from pathlib import Path

# TODO: Make these be args
EVALUATION_PATH = "evaluation"
INDICES = {"K": (3, 4), "B": (4, 11), "O": (4, 5)}
# INDICES = {"K": (3, 4), "B": (6, 11), "O": (4, 5)}
# INDICES = {"K": (3, 4), "B": (4, 5), "O": (4, 5)}
DIVIDE_BY = 1000000

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


def get_all_dataset_paths(eval_path: str) -> List[str]:
    return [
        f"{eval_path}/{system}/{dataset}"
        for system in listdir(f"{eval_path}/")
        for dataset in listdir(f"{eval_path}/{system}")
    ]


def parse_args():
    ArgumentParser()
    parser = ArgumentParser(description="Evaluate timing data for Monado visual-inertial tracking")
    parser.add_argument('timing_csv', type=Path, help="Timing file generated from Monado")
    parser.add_argument('gt_csv', type=Path, help="Groundtruth file for the dataset")

def main():
    stats = {}
    systems = listdir(EVALUATION_PATH)
    datasets = {ds for s in systems for ds in listdir(f"{EVALUATION_PATH}/{s}")}

    for dataset in datasets:
        stats[dataset] = {}
        for system in systems:
            timing_csv = f"{EVALUATION_PATH}/{system}/{dataset}/timing.csv"
            if not exists(timing_csv) or stat(timing_csv).st_size == 0:
                s = "—"
            else:
                i, j = INDICES[system[0]]
                s = stats_from_csv(timing_csv, i, j, DIVIDE_BY)
            stats[dataset][system] = s

    headers = ["Dataset"] + systems
    rows = [
        [d]
        + [
            f"{s.mean:.2f} ± {s.std:.2f}" if type(s) != str else s
            for ss, s in stats[d].items()
        ]
        for d in stats
    ]
    print(tabulate(rows, headers, tablefmt="pipe"))


if __name__ == "__main__":
    main()
