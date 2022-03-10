from typing import Tuple
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from tabulate import tabulate
from .utils import NS_TO, check_monotonic_rows


def parse_args():
    parser = ArgumentParser(
        description="Determine the tracking duration compared to the footage duration",
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="File generated from Monado (either tracking.csv or timing.csv)",
    )
    parser.add_argument(
        "frames_csv",
        type=Path,
        help="Dataset camera timestamp file (e.g. cam0/data.csv)",
    )
    parser.add_argument(
        "--units",
        type=str,
        help="Time units to show things on",
        default="s",
        choices=NS_TO.keys(),
    )
    return parser.parse_args()


def main():
    user_args = parse_args()
    input_csv = user_args.input_csv
    frames_csv = user_args.frames_csv
    units = user_args.units

    input_data = np.genfromtxt(input_csv, delimiter=",", comments="#", dtype=np.int64)
    frame_data = np.genfromtxt(frames_csv, delimiter=",", comments="#", dtype=np.int64)

    check_monotonic_rows(input_data)
    check_monotonic_rows(frame_data)

    itss = input_data[:, 0]
    ftss = frame_data[:, 0]

    i0, i1 = itss[0], itss[-1]
    f0, f1 = ftss[0], ftss[-1]

    nof_tracked_poses = itss.size
    nof_footage_frames = ftss.size
    tracking_duration = i1 - i0
    footage_duration = f1 - f0
    tracking_completion = tracking_duration / footage_duration

    table = [
        ["Input frames", f"{nof_footage_frames}"],
        ["Estimated poses", f"{nof_tracked_poses}"],
        ["Tracking duration", f"{tracking_duration / NS_TO[units]:.2f}{units}"],
        ["Footage duration", f"{footage_duration / NS_TO[units]:.2f}{units}"],
        ["Tracking completion", f"{tracking_completion * 100:.2f}%"],
    ]

    print(tabulate(table))  # tablefmt="pipe" for markdown


if __name__ == "__main__":
    main()
