import numpy as np
from collections import namedtuple
from argparse import ArgumentParser, Namespace
from pathlib import Path

from utils import NUMBER_OF_NS_IN, TIME_UNITS, load_trajectory

DEFAULT_TIME_UNITS = "ms"

TimingStats = namedtuple("TimingStats", ["mean", "std", "min", "q1", "q2", "q3", "max"])


def get_timing_stats(csv_fn: Path, i=0, j=-1, units=DEFAULT_TIME_UNITS) -> TimingStats:
    rows = load_trajectory(csv_fn)
    diffs = (rows[:, j] - rows[:, i]) / NUMBER_OF_NS_IN[units]
    return TimingStats(
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
        "--start_ts_idx",
        type=int,
        default=0,
        help="Column index of timing_csv to use as first timestamp",
    )
    parser.add_argument(
        "--end_ts_idx",
        type=int,
        default=-1,
        help="Column index of timing_csv to use as last timestamp",
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
    start_ts_idx = args.start_ts_idx
    end_ts_idx = args.end_ts_idx
    units = args.units
    s = get_timing_stats(csv_file, start_ts_idx, end_ts_idx, units)
    print(s)


if __name__ == "__main__":
    main()
