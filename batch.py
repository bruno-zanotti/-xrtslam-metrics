#!/usr/bin/env python

from argparse import ArgumentParser, RawTextHelpFormatter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import pandas as pd
import numpy as np
from tabulate import tabulate
from timing import TimingStats
from features import FeaturesStats
from completion import load_completion_stats
from tracking import get_tracking_stats
from utils import COMPLETION_FULL_SINCE, DEFAULT_TIMING_COLS, isnan, Vector2


@dataclass
class Batch:
    evaluation_path: Path
    targets_path: Path
    timing_columns: Dict[str, Tuple[str, str]]


SimpleMeasureF = Callable[[Path, str], Any]
TargetMeasureF = Callable[[Path, Path], Any]
MeasureFunction = Union[SimpleMeasureF, TargetMeasureF]
MeasureStringFunction = Callable[[Any], str]


def foreach_dataset(
    batch: Batch,
    result_fn: str,
    target_fn: Optional[str],
    measure,  #: MeasureFunction
    measure_str: MeasureStringFunction,
):
    sys_dirs = [r for r in batch.evaluation_path.iterdir() if r.is_dir()]
    ordered_set = {d.name: 0 for r in sys_dirs for d in r.iterdir() if d.is_dir()}
    ds_names = sorted(ordered_set.keys())
    sys_names = sorted([d.name for d in sys_dirs])
    df = pd.DataFrame(None, columns=sys_names, index=ds_names)

    for sys_dir in sys_dirs:
        for ds_name in ds_names:
            ds_dir = sys_dir / ds_name
            if ds_dir.is_dir():
                sys_name = sys_dir.name
                result_csv = ds_dir / result_fn
                if result_csv.exists() and result_csv.stat().st_size != 0:
                    if target_fn is None:
                        df[sys_name][ds_name] = measure(result_csv, sys_name)
                    else:
                        target_csv = batch.targets_path / ds_name / target_fn
                        if target_csv.exists():
                            df[sys_name][ds_name] = measure(result_csv, target_csv)

    # Add average row
    avg_row = df.apply(lambda col: col.mean(), result_type="reduce")
    new_index = df.shape[0]
    df.loc[new_index] = avg_row  # type: ignore
    df = df.rename({new_index: "[AVG]"})
    measure_str_none = lambda m: "—" if isnan(m) else measure_str(m)
    df_to_string = df.applymap(measure_str_none)
    print(tabulate(df_to_string, headers="keys", tablefmt="pipe"))  # type: ignore


def timing_main(batch: Batch):
    print("\nAverage (± stdev) pose estimation time [ms]\n")

    def measure_timing(result_csv: Path, sys_name: str) -> Vector2:
        cols = batch.timing_columns.get(sys_name, DEFAULT_TIMING_COLS)
        s = TimingStats(csv_fn=result_csv, cols=cols)
        return np.array([s.mean, s.std])

    def measure_timing_str(measure: Vector2) -> str:
        mean, std = measure
        return f"{mean:.2f} ± {std:.2f}"

    foreach_dataset(batch, "timing.csv", None, measure_timing, measure_timing_str)


def features_main(batch: Batch):
    print("\nAverage feature count for each camera\n")

    def measure_features(result_csv: Path, _: str) -> Vector2:
        s = FeaturesStats(csv_fn=result_csv)
        return np.array([s.mean, s.std])

    def measure_features_str(measure: Vector2) -> str:
        mean, std = measure
        return f"{mean.astype(int)} ± {std.astype(int)}"

    foreach_dataset(batch, "features.csv", None, measure_features, measure_features_str)


def completion_main(batch: Batch):
    print("\nAverage completion percentage [%]\n")

    def measure_completion(result_csv: Path, target_csv: Path) -> float:
        s = load_completion_stats(result_csv, target_csv)
        return s.tracking_completion

    def measure_completion_str(completion: float) -> str:
        return f"{completion * 100:.2f}%" if completion < COMPLETION_FULL_SINCE else "✓"

    foreach_dataset(
        batch, "tracking.csv", "cam0.csv", measure_completion, measure_completion_str
    )


def ate_main(batch: Batch):
    print("\nAbsolute trajectory error (ATE) [m]\n")

    def measure_ape(result_csv: Path, target_csv: Path) -> Vector2:
        results = get_tracking_stats("ate", [result_csv], target_csv, silence=True)
        s = results[result_csv].stats
        # Notice that std runs over APE while rmse over APE²
        return np.array([s["rmse"], s["std"]])

    def measure_ape_str(measure: Vector2) -> str:
        rmse, std = measure
        return f"{rmse:.3f} ± {std:.3f}"

    foreach_dataset(batch, "tracking.csv", "gt.csv", measure_ape, measure_ape_str)


def rte_main(batch: Batch):
    print("\nRelative trajectory error (RTE) [m]\n")

    def measure_rpe(result_csv: Path, target_csv: Path) -> Vector2:
        results = get_tracking_stats("rte", [result_csv], target_csv, silence=True)
        s = results[result_csv].stats
        # Notice that std runs over RPE while rmse over RPE²
        return np.array([s["rmse"], s["std"]])

    def measure_rpe_str(measure: Vector2) -> str:
        rmse, std = measure
        return f"{rmse:.6f} ± {std:.6f}"

    foreach_dataset(batch, "tracking.csv", "gt.csv", measure_rpe, measure_rpe_str)


def seg_main(batch: Batch):
    # TODO: Segment drift per second (SDS) second might be seful
    # TODO: Get SDM tolerance programatically instead of hardcoding 1cm
    # TODO: Does SD tolerance affect a lot the final numbers in SDM/SDS?
    print("\n Segment drift per meter error (SDM 0.01m) [m/m]\n")

    def measure_seg(result_csv: Path, target_csv: Path) -> Vector2:
        results = get_tracking_stats("seg", [result_csv], target_csv, silence=True)
        s = results[result_csv].stats
        return np.array([s["SDM"], s["SDM std"]])

    def measure_seg_str(measure: Vector2) -> str:
        drift, std = measure
        return f"{drift:.4f} ± {std:.4f}"

    foreach_dataset(batch, "tracking.csv", "gt.csv", measure_seg, measure_seg_str)


def parse_args():
    parser = ArgumentParser(
        description="Batch evaluation of Monado visual-inertial runs of datasets\n\n"
        "Example execution: \n"
        "python batch.py test/data/runs/ test/data/targets/ \\ \n"
        "\t--timing Basalt opticalflow_received vio_produced \\\n"
        "\t--timing Kimera tracker_pushed processed \\\n"
        "\t--timing ORB-SLAM3 about_to_process processed",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "runs_dir",
        type=Path,
        help="Directory with runs subdirectories, each with datasets subdirectories."
        "The structure of runs_dir is like: <runs_dir>/<run>/<dataset>/{tracking, timing}.csv",
    )
    parser.add_argument(
        "targets_dir",
        type=Path,
        help="Directory with dataset groundtruth and camera timestamps."
        "The structure of targets_dir is like: <targets_dir>/<dataset>/{gt, cam0}.csv",
    )
    parser.add_argument(
        "--timing",
        action="append",
        nargs=3,
        default=[],
        help="For each <run> directory in <runs_dir> specify the first and last"
        "timing column names to use as --timing <run> <first_col> <last_col>."
        "If a <run> is not specified assuming"
        f"<first_col> = {DEFAULT_TIMING_COLS[0]} and <last_col> = {DEFAULT_TIMING_COLS[1]}",
    )
    return parser.parse_args()


def batch_from_args(args) -> Batch:
    timing_columns = {}
    for run, first_col, last_col in args.timing:
        timing_columns[run] = (first_col, last_col)
    batch = Batch(args.runs_dir, args.targets_dir, timing_columns)
    return batch


def main():
    batch = batch_from_args(parse_args())
    timing_main(batch)
    features_main(batch)
    completion_main(batch)
    ate_main(batch)
    rte_main(batch)
    seg_main(batch)


if __name__ == "__main__":
    main()
