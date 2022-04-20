#!/usr/bin/env python

from typing import Tuple
from pathlib import Path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from evo.tools import file_interface, plot
from evo.tools.plot import PlotMode
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation, Unit

from completion import CompletionStats


def parse_args():
    parser = ArgumentParser(
        description="Determine absolute pose error for a trajectory and its groundtruth",
    )
    parser.add_argument(
        "metric",
        type=str,
        help="What tracking metric to compute",
        default="ate",
        choices=["ate", "rte"],
    )
    parser.add_argument(
        "tracking_csv",
        type=Path,
        help="Tracking file generated from Monado",
    )
    parser.add_argument(
        "groundtruth_csv",
        type=Path,
        help="Dataset groundtruth file",
    )
    parser.add_argument(
        "-p",
        "--plot",
        help="Enable to show trajectory plot",
        action="store_true",
    )
    return parser.parse_args()


def get_sanitized_trajectories(
    tracking_csv: Path, groundtruth_csv: Path
) -> Tuple[PoseTrajectory3D, PoseTrajectory3D, CompletionStats]:
    """Trim and synchronizes trajectories so that they have the same amount of poses"""
    # NOTE: Evo uses doubles for its timestamps and thus looses a bit of
    # precision, but even in the worst case, the precision is about ~1usec
    traj_ref = file_interface.read_euroc_csv_trajectory(groundtruth_csv)
    traj_est = file_interface.read_euroc_csv_trajectory(tracking_csv)

    # Trim both trajectories so that only overlapping timestamps are kept
    e0, e1 = traj_est.timestamps[0], traj_est.timestamps[-1]
    r0, r1 = traj_ref.timestamps[0], traj_ref.timestamps[-1]
    first_ts = max(e0, r0)
    last_ts = min(e1, r1)
    traj_ref.reduce_to_time_range(first_ts, last_ts)
    traj_est.reduce_to_time_range(first_ts, last_ts)

    c = CompletionStats(
        e0, e1, r0, r1, traj_est.timestamps.size, traj_ref.timestamps.size
    )

    # TODO: PR with a more realtime-appropriate trajectory alignment.
    # `associate_trajectories`` synchronizes the two trajectories as follows:
    # 1. The trajectory with less poses is kept
    # 2. In the second trajectory only the poses with closest timestamps to the
    #    first trajectory are kept.
    # A way of syncing trajectories a tad more meaningful for VR would be to
    # always use the previously tracked pose for each groundtruth pose.
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    return traj_est, traj_ref, c


def get_tracking_ape_stats(
    tracking_csv: Path,
    groundtruth_csv: Path,
    pose_relation: PoseRelation = PoseRelation.translation_part,
    show_plot: bool = False,
):
    traj_est, traj_ref, c = get_sanitized_trajectories(tracking_csv, groundtruth_csv)
    ref_name = "reference"
    est_name = "estimated"
    umemaya_align = True  # Otherwise align_origin
    result = main_ape.ape(
        traj_ref=traj_ref,
        traj_est=traj_est,
        pose_relation=pose_relation,  # TODO: I might want to also consider rotation_angle_deg
        align=umemaya_align,
        correct_scale=False,
        n_to_align=-1,
        align_origin=umemaya_align,  # TODO: Doesn't this make a lot of sense for VR?
        ref_name=ref_name,
        est_name=est_name,
    )

    if show_plot:
        fig = plt.figure()
        plot_mode = PlotMode.xyz
        ax = plot.prepare_axis(fig, plot_mode)
        plot.traj(ax, plot_mode, traj_ref, style="--", alpha=0.5)
        plot.traj_colormap(
            ax,
            result.trajectories[est_name],
            result.np_arrays["error_array"],
            plot_mode,
            min_map=result.stats["min"],
            max_map=result.stats["max"],
        )
        plt.show()

    return result, c


def get_tracking_rpe_stats(
    tracking_csv: Path,
    groundtruth_csv: Path,
    pose_relation: PoseRelation = PoseRelation.translation_part,
    show_plot: bool = False,
):
    traj_est, traj_ref, c = get_sanitized_trajectories(tracking_csv, groundtruth_csv)
    ref_name = "reference"
    est_name = "estimated"
    umemaya_align = True  # Otherwise align_origin
    result = main_rpe.rpe(
        traj_ref=traj_ref,
        traj_est=traj_est,
        pose_relation=pose_relation,  # TODO: I might want to also consider rotation_angle_deg? ROE?
        delta=6,
        delta_unit=Unit.frames,  # TODO: Evo doesn't support delta_units=seconds, I think 0.2s would be good
        rel_delta_tol=0.1,  # TODO: This only seems to be used when all_pairs is enabled
        all_pairs=False,  # TODO: Use all_pairs?
        align=umemaya_align,
        correct_scale=False,
        n_to_align=-1,
        align_origin=umemaya_align,  # TODO: Doesn't this make a lot of sense for VR?
        ref_name=ref_name,
        est_name=est_name,
        support_loop=False,  # Seems to only be used to not modify the input trajectories in jupyter notebooks
    )

    if show_plot:
        fig = plt.figure()
        plot_mode = PlotMode.xyz
        ax = plot.prepare_axis(fig, plot_mode)
        plot.traj(ax, plot_mode, traj_ref, style="--", alpha=0.5)
        plot.traj_colormap(
            ax,
            result.trajectories[est_name],
            result.np_arrays["error_array"],
            plot_mode,
            min_map=result.stats["min"],
            max_map=result.stats["max"],
        )
        plt.show()

    return result, c


def main():
    args = parse_args()
    metric = args.metric
    tracking_csv = args.tracking_csv
    groundtruth_csv = args.groundtruth_csv
    show_plot = args.plot

    if metric == "ate":
        result, completion = get_tracking_ape_stats(
            tracking_csv, groundtruth_csv, show_plot=show_plot
        )
    elif metric == "rte":
        result, completion = get_tracking_rpe_stats(
            tracking_csv, groundtruth_csv, show_plot=show_plot
        )
    else:
        assert False

    print(result)
    # print("\nCompletion:")
    # print(completion)

    # TODO: Try to reuse EVO settings as much as possible
    # TODO: Bring position/rpy graphs as well like in evo_traj


if __name__ == "__main__":
    main()
