#!/usr/bin/env python

from typing import Callable, Optional, Union
from tabulate import tabulate
from timing import load_timing_stats
from completion import load_completion_stats
from tracking import get_tracking_ape_stats, get_tracking_rpe_stats
from pathlib import Path
import pandas as pd

# TODO: Use argparse
DIVIDE_BY = 1000000  # TODO: Use --units (make it a parent argparser)
EVALUATION_PATH = Path("runs")
# EVALUATION_PATH = Path("runs-max-speed")
TARGETS_PATH = Path("targets")
INDICES = {"K": (3, 4), "B": (4, 11), "O": (4, 5)}
UNITS = "ms"

SimpleMeasureF = Callable[[Path, str], str]
TargetMeasureF = Callable[[Path, Path], str]
MeasureFunction = Union[SimpleMeasureF, TargetMeasureF]


def foreach_dataset(
    result_fn: str, target_fn: Optional[str], measure_str  #: MeasureFunction
):
    sys_dirs = [r for r in EVALUATION_PATH.iterdir() if r.is_dir()]
    ordered_set = {d.name: 0 for r in sys_dirs for d in r.iterdir() if d.is_dir()}
    ds_names = sorted(ordered_set.keys())
    sys_names = sorted([d.name for d in sys_dirs])
    df = pd.DataFrame("—", columns=sys_names, index=ds_names)

    for sys_dir in sys_dirs:
        for ds_name in ds_names:
            ds_dir = sys_dir / ds_name
            if ds_dir.is_dir():
                sys_name = sys_dir.name
                result_csv = ds_dir / result_fn
                if result_csv.exists() and result_csv.stat().st_size != 0:
                    if target_fn is None:
                        df[sys_name][ds_name] = measure_str(result_csv, sys_name)
                    else:
                        target_csv = TARGETS_PATH / ds_name / target_fn
                        if target_csv.exists():
                            df[sys_name][ds_name] = measure_str(result_csv, target_csv)

    print(tabulate(df, headers="keys", tablefmt="pipe"))


def timing_main():
    def measure_timing(result_csv: Path, sys_name: str) -> str:
        i, j = INDICES[sys_name[0]]
        s = load_timing_stats(result_csv, i=i, j=j)
        return f"{s.mean:.2f} ± {s.std:.2f}"

    foreach_dataset("timing.csv", None, measure_timing)


def completion_main():
    def measure_completion(result_csv: Path, target_csv: Path) -> str:
        s = load_completion_stats(result_csv, target_csv)
        r = (
            f"{s.tracking_completion * 100:.2f}%"
            if s.tracking_completion < 0.98
            else "✓"
        )
        return r

    foreach_dataset("tracking.csv", "cam0.csv", measure_completion)


def ate_main():
    def measure_ape(result_csv: Path, target_csv: Path) -> str:
        s = get_tracking_ape_stats(result_csv, target_csv)[0].stats
        return f"{s['mean']:.3f} ± {s['std']:.3f}"

    foreach_dataset("tracking.csv", "gt.csv", measure_ape)


def rte_main():
    def measure_rpe(result_csv: Path, target_csv: Path) -> str:
        s = get_tracking_rpe_stats(result_csv, target_csv)[0].stats
        return f"{s['mean']:.3f} ± {s['std']:.3f}"

    foreach_dataset("tracking.csv", "gt.csv", measure_rpe)


def main():
    timing_main()
    completion_main()
    ate_main()
    rte_main()


if __name__ == "__main__":
    main()
