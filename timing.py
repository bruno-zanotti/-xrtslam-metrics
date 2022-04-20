#!/usr/bin/env python

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from utils import COLORS, NUMBER_OF_NS_IN, TIME_UNITS, DEFAULT_TIME_UNITS, load_timing


class TimingStats:
    def __init__(
        self,
        csv_fn: Optional[Path] = None,
        column_names: Optional[List[str]] = None,
        timing_data: Optional[np.ndarray] = None,
        cols: Optional[Tuple[str, str]] = None,
        units: str = DEFAULT_TIME_UNITS,
    ):
        self.units = units
        if column_names and timing_data:
            self.column_names = column_names
            self.timing_data = timing_data
        elif csv_fn:
            self.column_names, self.timing_data = load_timing(csv_fn)
        else:
            raise Exception(f"Invalid parameters for {TimingStats.__name__}")

        # silence mypy
        assert self.column_names is not None and self.timing_data is not None

        assert (
            len(self.column_names) == self.timing_data.shape[1]
        ), "column names differ from data columns"

        if cols:
            self.set_cols(*cols)

    def set_cols(self, first: str, last: str):
        assert (
            first in self.column_names and last in self.column_names
        ), f"columns '{first}' or '{last}' not in {self.column_names=}"
        self.first_column = first
        self.last_column = last
        self.i = self.column_names.index(first)
        self.j = self.column_names.index(last)

        self.diffs = (
            self.timing_data[:, self.j] - self.timing_data[:, self.i]
        ) / NUMBER_OF_NS_IN[self.units]

    @property
    def mean(self):
        return np.mean(self.diffs)

    @property
    def std(self):
        return np.std(self.diffs)

    @property
    def min(self):
        return np.min(self.diffs)

    @property
    def q1(self):
        return np.quantile(self.diffs, 0.25)

    @property
    def q2(self):
        return np.median(self.diffs)

    @property
    def q3(self):
        return np.quantile(self.diffs, 0.75)

    @property
    def max(self):
        return np.max(self.diffs)

    def __str__(self) -> str:
        return f"TimingStats(mean={self.mean}, std={self.std}, min={self.min}, q1={self.q1}, q2={self.q2}, q3={self.q3}, max={self.max}) from '{self.first_column}' to '{self.last_column}'"

    def plot(self) -> None:
        td = self.timing_data

        framepose_tss = (td[:, 0] - td[0, 0]) / NUMBER_OF_NS_IN["s"]

        td1 = np.concatenate((td[:, 0].reshape(td.shape[0], 1), td[:, :-1]), axis=1)
        td2 = td - td1
        assert not td2[:, 0].any(), "first column should be zeroed out by this point"
        # assert td2[123, 7] == td[123, 7] - td[123, 6] # random check that should pass
        td3 = td2 / NUMBER_OF_NS_IN[self.units]

        deltas_by_column = [td3[:, i] for i, cn in enumerate(self.column_names)]

        labels = [
            f"{i+1}. {cn}"
            for i, cn in enumerate(self.column_names[self.i : self.j + 1])
        ]

        _, ax = plt.subplots()
        ax.stackplot(
            framepose_tss,
            deltas_by_column[self.i : self.j + 1],
            labels=labels,
            colors=COLORS,
            alpha=0.8,
        )
        ax.legend(loc="upper right")
        ax.set_title("Stacked timing")
        ax.set_xlabel("Dataset time (s)")
        ax.set_ylabel("Processing duration (ms)")

        plt.show()


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
        "-p",
        "--plot",
        help="Whether to plot a stacked timing graph",
        action="store_true",
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
    args = parse_args()
    csv_file = args.timing_csv
    first_column = args.first_column
    last_column = args.last_column
    plot = args.plot
    units = args.units
    s = TimingStats(csv_fn=csv_file, cols=(first_column, last_column), units=units)
    print(s)

    if plot:
        s.plot()


if __name__ == "__main__":
    main()
