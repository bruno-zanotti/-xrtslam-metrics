#!/usr/bin/env python

import numpy as np
from collections import namedtuple
from argparse import ArgumentParser, Namespace
from pathlib import Path

from utils import NUMBER_OF_NS_IN, TIME_UNITS, load_timing

DEFAULT_TIME_UNITS = "ms"

TimingStats = namedtuple("TimingStats", ["mean", "std", "min", "q1", "q2", "q3", "max"])


def get_timing_stats(timing_data: np.ndarray, i: int, j: int, units=DEFAULT_TIME_UNITS):
    diffs = (timing_data[:, j] - timing_data[:, i]) / NUMBER_OF_NS_IN[units]
    return TimingStats(
        np.mean(diffs),
        np.std(diffs),
        np.min(diffs),
        np.quantile(diffs, 0.25),
        np.median(diffs),
        np.quantile(diffs, 0.75),
        np.max(diffs),
    )


def load_timing_stats(
    csv_fn: Path, col1: str, col2: str, units=DEFAULT_TIME_UNITS
) -> TimingStats:
    column_names, timing_data = load_timing(csv_fn)

    assert (
        col1 in column_names and col2 in column_names
    ), f"columns '{col1}' or '{col2}' not in {column_names=}"

    i, j = column_names.index(col1), column_names.index(col2)
    return get_timing_stats(timing_data, i, j, units)


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
        "first_column",
        type=str,
        # default="tracker_receiving_left_frame", TODO: Standardize a couple of column names
        help="Column name of timing_csv to use as first timestamp",
    )
    parser.add_argument(
        "last_column",
        type=str,
        # default="tracker_processed_pose", TODO: Standardize a couple of column names
        help="Column name of timing_csv to use as last timestamp",
    )
    parser.add_argument(
        "--units",
        type=str,
        help="Time units to show things on",
        default=DEFAULT_TIME_UNITS,
        choices=TIME_UNITS,
    )
    return parser.parse_args()


def main():
    global args
    args = parse_args()
    csv_file = args.timing_csv
    first_column = args.first_column
    last_column = args.last_column
    units = args.units
    s = load_timing_stats(csv_file, first_column, last_column, units)
    print(s)


if __name__ == "__main__":
    main()
