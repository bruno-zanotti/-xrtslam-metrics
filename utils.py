from pathlib import Path
from typing import List, Tuple

import numpy as np

NUMBER_OF_NS_IN = {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}
TIME_UNITS = NUMBER_OF_NS_IN.keys()
DEFAULT_TIME_UNITS = "ms"
COMPLETION_FULL_SINCE = 0.98  # Completion ratio to consider a complete run
DEFAULT_TIMING_COLS = ("frames_received", "pose_produced")
COLORS = [  # Regular cycle
    "#2196F3",  # blue
    "#4CAF50",  # green
    "#FFC107",  # amber
    "#E91E63",  # pink
    "#673AB7",  # deeppurple
    "#00BCD4",  # cyan
    "#CDDC39",  # lime
    "#FF5722",  # deeporange
    "#9C27B0",  # purple
    "#03A9F4",  # lightblue
    "#8BC34A",  # lightgreen
    "#FF9800",  # orange
    "#3F51B5",  # indigo
    "#009688",  # teal
    "#FFEB3B",  # yellow
    "#F44336",  # red
    "#795548",  # brown
    "#607D8B",  # bluegrey
    "#546E7A",  # shade of gray
    "#455A64",  # shade of gray
    "#37474F",  # shade of gray
    "#263238",  # shade of gray
    "#212121",  # shade of gray
] + ["#000000"] * 100

make_color_iterator = lambda: (c for c in COLORS)


def check_monotonic_rows(rows: np.ndarray) -> None:
    last_ts = 0
    for row in rows:
        ts = row[0]
        assert last_ts < ts
        last_ts = ts


def load_csv(csv_fn: Path, dtype=np.int64) -> np.ndarray:
    data = np.genfromtxt(
        csv_fn, delimiter=",", comments="#", dtype=dtype, invalid_raise=False
    )
    check_monotonic_rows(data)
    return data


def load_trajectory(csv_fn: Path, dtype=np.int64) -> np.ndarray:
    # TODO: Add unit quaternion check
    return load_csv(csv_fn, dtype)


def load_csv_safer(csv_fn: Path, dtype=np.int64) -> Tuple[List[str], np.ndarray]:
    timing_data = load_csv(csv_fn, dtype)

    with open(csv_fn, "r", encoding="utf8") as f:
        first_line = next(f)
    assert (
        first_line[0] == "#" and first_line[-1] == "\n"
    ), "first csv line should be a comment with column names"

    column_names = first_line[1:-1].split(",")
    assert (
        len(column_names) == timing_data.shape[1]
    ), "number of column names differ from data columns"

    return column_names, timing_data


# Logging utils


def color_string(string, fg=None):
    ANSI_COLORS = {  # List some colors that may be needed
        "red": "\033[31m",
        "pink": "\033[38;5;206m",
        "green": "\033[32m",
        "orange": "\033[33m",
        "blue": "\033[34m",
        "cyan": "\033[36m",
        "lightred": "\033[91m",
        "lightgreen": "\033[92m",
        "yellow": "\033[93m",
        "lightblue": "\033[94m",
        "lightcyan": "\033[96m",
        "brightwhite": "\u001b[37;1m",
        "brightmagenta": "\u001b[35;1m",
    }
    endcolor = "\033[0m"
    return f"{ANSI_COLORS.get(fg, '')}{string}{endcolor}"


def info(*lines):
    for line in lines:
        print(f"{color_string('[I] ', fg='cyan')}{line}")


def error(*lines, should_exit=True):
    for line in lines:
        print(f"{color_string('[E] ', fg='lightred')}{line}")
    if should_exit:
        exit(1)


def warn(*lines):
    for line in lines:
        print(f"{color_string('[W] ', fg='yellow')}{line}")
