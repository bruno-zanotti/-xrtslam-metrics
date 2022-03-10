from argparse import ArgumentParser
import numpy as np
from pathlib import Path

NUMBER_OF_NS_IN = {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}
TIME_UNITS = NUMBER_OF_NS_IN.keys()


def check_monotonic_rows(rows: np.ndarray) -> None:
    last_ts = 0
    for row in rows:
        ts = row[0]
        assert last_ts < ts
        last_ts = ts


def load_trajectory(csv_fn: Path, dtype=np.int64) -> np.ndarray:
    data = np.genfromtxt(csv_fn, delimiter=",", comments="#", dtype=dtype)
    check_monotonic_rows(data)
    # TODO: Add unit quaternion check
    return data
