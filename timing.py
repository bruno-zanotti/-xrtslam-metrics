import numpy as np
from collections import namedtuple
from argparse import ArgumentParser, Namespace
from pathlib import Path

from utils import NUMBER_OF_NS_IN, TIME_UNITS, load_trajectory

args: Namespace  # Arguments from CLI

TimingStats = namedtuple("TimingStats", ["mean", "std", "min", "q1", "q2", "q3", "max"])


def get_timing_stats(csv_fn: Path, i=0, j=-1, units=None) -> TimingStats:
    units = args.units if not units else units
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
        "--units",
        type=str,
        help="Time units to show things on",
        default="ms",
        choices=TIME_UNITS,
    )
    return parser.parse_args()


def main():
    global args
    args = parse_args()
    csv_file = args.timing_csv
    units = args.units
    s = get_timing_stats(csv_file, units=units)
    print(s)


if __name__ == "__main__":
    main()
