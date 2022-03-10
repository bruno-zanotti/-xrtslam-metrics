import csv
from typing import List
from os import listdir, stat
from os.path import exists
from tabulate import tabulate
from timing import get_timing_stats

# TODO: Use argparse
DIVIDE_BY = 1000000  # TODO: Use --units (make it a parent argparser)
EVALUATION_PATH = "evaluation"
INDICES = {"K": (3, 4), "B": (4, 11), "O": (4, 5)}
UNITS = "ms"


def get_all_dataset_paths(eval_path: str) -> List[str]:
    return [
        f"{eval_path}/{system}/{dataset}"
        for system in listdir(f"{eval_path}/")
        for dataset in listdir(f"{eval_path}/{system}")
    ]


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
                s = get_timing_stats(timing_csv, i, j, UNITS)
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
