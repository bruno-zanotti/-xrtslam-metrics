from collections import namedtuple
from typing import Tuple
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tabulate import tabulate
from utils import TIME_UNITS, NUMBER_OF_NS_IN, load_trajectory
from dataclasses import dataclass

args: Namespace  # Arguments from CLI


@dataclass
class CompletionStats:
    first_tracked_ts: int
    last_tracked_ts: int
    first_gt_ts: int
    last_gt_ts: int
    nof_tracked_poses: int
    nof_gt_poses: int

    @property
    def tracking_duration(self):
        return self.last_tracked_ts - self.first_tracked_ts

    @property
    def gt_duration(self):
        return self.last_gt_ts - self.first_gt_ts

    @property
    def tracking_completion(self):
        return self.tracking_duration / self.gt_duration

    def __str__(self):
        units = args.units
        div = NUMBER_OF_NS_IN[units]

        table = [
            ["Groundtruth poses", f"{self.nof_gt_poses}"],
            ["Estimated poses", f"{self.nof_tracked_poses}"],
            ["Tracking duration", f"{self.tracking_duration / div:.2f}{units}"],
            ["Groundtruth duration", f"{self.gt_duration / div:.2f}{units}"],
            ["Tracking completion", f"{self.tracking_completion * 100:.2f}%"],
        ]
        return tabulate(table, tablefmt="pipe")


def parse_args():
    parser = ArgumentParser(
        description="Determine information about tracking completion based on groundtruth duration"
    )
    parser.add_argument(
        "tracking_csv",
        type=Path,
        help="File generated from Monado (either tracking.csv or timing.csv)",
    )
    parser.add_argument(
        "groundtruth_csv",
        type=Path,
        help="Dataset groundtruth file",
    )
    parser.add_argument(
        "--units",
        type=str,
        help="Time units to show things on",
        default="s",
        choices=TIME_UNITS,
    )
    return parser.parse_args()


def get_completion_stats(tracking_csv: Path, gt_csv: Path) -> CompletionStats:
    tdata = load_trajectory(tracking_csv)
    gdata = load_trajectory(gt_csv)

    ttss = tdata[:, 0]
    gtss = gdata[:, 0]

    t0, t1 = ttss[0], ttss[-1]
    g0, g1 = gtss[0], gtss[-1]

    stats = CompletionStats(t0, t1, g0, g1, ttss.size, gtss.size)
    return stats


def main():
    global args
    args = parse_args()
    tracking_csv = args.tracking_csv
    groundtruth_csv = args.groundtruth_csv
    units = args.units

    s = get_completion_stats(tracking_csv, groundtruth_csv)

    print(str(s))


if __name__ == "__main__":
    main()
