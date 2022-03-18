from typing import List, Tuple
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
    data = np.genfromtxt(
        csv_fn, delimiter=",", comments="#", dtype=dtype, invalid_raise=False
    )
    check_monotonic_rows(data)
    # TODO: Add unit quaternion check
    return data


def load_timing(csv_fn: Path, dtype=np.int64) -> Tuple[List[str], np.ndarray]:
    timing_data = load_trajectory(csv_fn, dtype)

    with open(csv_fn, "r") as f:
        first_line = next(f)
    assert (
        first_line[0] == "#" and first_line[-1] == "\n"
    ), "first csv line should be a comment with column names"

    column_names = first_line[1:-1].split(",")
    assert (
        len(column_names) == timing_data.shape[1]
    ), "column names differ from data columns"

    return column_names, timing_data
