#!/usr/bin/env python

from argparse import ArgumentParser
from pathlib import Path
from typing import List

import numpy as np

from utils import load_csv_safer


class FeaturesStats:
    def __init__(self, csv_fn: Path):
        self.column_names, self.features_data = load_csv_safer(csv_fn)
        self.nof_features = self.features_data[:, 1:3]

    @property
    def mean(self):
        return np.mean(self.nof_features, axis=0)

    @property
    def std(self):
        return np.std(self.nof_features, axis=0)

    @property
    def min(self):
        return np.min(self.nof_features, axis=0)

    @property
    def q1(self):
        return np.quantile(self.nof_features, 0.25, axis=0)

    @property
    def q2(self):
        return np.median(self.nof_features, axis=0)

    @property
    def q3(self):
        return np.quantile(self.nof_features, 0.75, axis=0)

    @property
    def max(self):
        return np.max(self.nof_features, axis=0)

    def __str__(self) -> str:
        return f"FeaturesStats(mean={self.mean}, std={self.std}, min={self.min}, q1={self.q1}, q2={self.q2}, q3={self.q3}, max={self.max})"


def parse_args():
    parser = ArgumentParser(
        description="Measure visual features metrics for Monado visual-inertial tracking",
    )
    parser.add_argument(
        "features_csv",
        type=Path,
        help="Features file generated from Monado",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_file = args.features_csv
    s = FeaturesStats(csv_fn=csv_file)
    print(s)

    # if plot: # TODO: Plot multiple features
    #     s.plot()


if __name__ == "__main__":
    main()
