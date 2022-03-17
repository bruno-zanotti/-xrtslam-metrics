#!/usr/bin/env python

import csv
from typing import List
from os import listdir, stat
from os.path import exists
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
GT_PATH = Path("gts")
INDICES = {"K": (3, 4), "B": (4, 11), "O": (4, 5)}
UNITS = "ms"


def get_all_dataset_paths(eval_path: str) -> List[str]:
    return [
        f"{eval_path}/{system}/{dataset}"
        for system in listdir(f"{eval_path}/")
        for dataset in listdir(f"{eval_path}/{system}")
    ]


def timing_main():
    EXPECTED_CSV_FILE = "timing.csv"
    sys_dirs = [r for r in EVALUATION_PATH.iterdir() if r.is_dir()]
    ds_names = {d.name: 0 for r in sys_dirs for d in r.iterdir() if d.is_dir()}
    ds_names = sorted(ds_names.keys())
    sys_names = sorted([d.name for d in sys_dirs])
    df = pd.DataFrame("—", columns=sys_names, index=ds_names)

    for sys_dir in sys_dirs:
        for ds_name in ds_names:
            ds_dir = sys_dir / ds_name
            if ds_dir.is_dir():
                sys_name = sys_dir.name
                csvfile = ds_dir / EXPECTED_CSV_FILE
                if csvfile.exists() and csvfile.stat().st_size != 0:
                    ### TODO: Generalize the rest
                    i, j = INDICES[sys_name[0]]
                    s = load_timing_stats(csvfile, i=i, j=j)
                    ### TODO: Generalize the rest
                    df[sys_name][ds_name] = f"{s.mean:.2f} ± {s.std:.2f}"

    print(tabulate(df, headers="keys"))


def completion_main():
    EXPECTED_CSV_FILE = "tracking.csv"
    sys_dirs = [r for r in EVALUATION_PATH.iterdir() if r.is_dir()]
    ds_names = {d.name: 0 for r in sys_dirs for d in r.iterdir() if d.is_dir()}
    ds_names = sorted(ds_names.keys())
    sys_names = sorted([d.name for d in sys_dirs])
    df = pd.DataFrame("—", columns=sys_names, index=ds_names)

    for sys_dir in sys_dirs:
        for ds_name in ds_names:
            ds_dir = sys_dir / ds_name
            if ds_dir.is_dir():
                sys_name = sys_dir.name
                csvfile = ds_dir / EXPECTED_CSV_FILE
                if csvfile.exists() and csvfile.stat().st_size != 0:
                    ### TODO: Generalize the rest
                    frames_csv = GT_PATH / ds_name / "cam0.csv"
                    if frames_csv.exists():
                        s = load_completion_stats(csvfile, frames_csv)
                        ### TODO: Generalize the rest
                        # df[sys_name][ds_name] = f"{s.tracking_completion * 100:.2f}%"
                        df[sys_name][ds_name] = (
                            f"{s.tracking_completion * 100:.2f}%"
                            if s.tracking_completion < 0.98
                            else "✓"
                        )
    print(tabulate(df, headers="keys"))


def ate_main():
    EXPECTED_CSV_FILE = "tracking.csv"
    sys_dirs = [r for r in EVALUATION_PATH.iterdir() if r.is_dir()]
    ds_names = {d.name: 0 for r in sys_dirs for d in r.iterdir() if d.is_dir()}
    ds_names = sorted(ds_names.keys())
    sys_names = sorted([d.name for d in sys_dirs])
    df = pd.DataFrame("—", columns=sys_names, index=ds_names)

    for sys_dir in sys_dirs:
        for ds_name in ds_names:
            ds_dir = sys_dir / ds_name
            if ds_dir.is_dir():
                sys_name = sys_dir.name
                csvfile = ds_dir / EXPECTED_CSV_FILE
                if csvfile.exists() and csvfile.stat().st_size != 0:
                    ### TODO: Generalize the rest
                    gt_csv = GT_PATH / ds_name / "gt.csv"
                    if gt_csv.exists():
                        s = get_tracking_ape_stats(csvfile, gt_csv)[0].stats
                        ### TODO: Generalize the rest
                        df[sys_name][ds_name] = f"{s['mean']:.3f} ± {s['std']:.3f}"

    print(tabulate(df, headers="keys"))

def rte_main():
    EXPECTED_CSV_FILE = "tracking.csv"
    sys_dirs = [r for r in EVALUATION_PATH.iterdir() if r.is_dir()]
    ds_names = {d.name: 0 for r in sys_dirs for d in r.iterdir() if d.is_dir()}
    ds_names = sorted(ds_names.keys())
    sys_names = sorted([d.name for d in sys_dirs])
    df = pd.DataFrame("—", columns=sys_names, index=ds_names)

    for sys_dir in sys_dirs:
        for ds_name in ds_names:
            ds_dir = sys_dir / ds_name
            if ds_dir.is_dir():
                sys_name = sys_dir.name
                csvfile = ds_dir / EXPECTED_CSV_FILE
                if csvfile.exists() and csvfile.stat().st_size != 0:
                    ### TODO: Generalize the rest
                    gt_csv = GT_PATH / ds_name / "gt.csv"
                    if gt_csv.exists():
                        s = get_tracking_rpe_stats(csvfile, gt_csv)[0].stats
                        ### TODO: Generalize the rest
                        df[sys_name][ds_name] = f"{s['mean']:.3f} ± {s['std']:.3f}"

    print(tabulate(df, headers="keys"))


def main():
    timing_main()
    completion_main()
    ate_main()
    rte_main()


if __name__ == "__main__":
    main()
