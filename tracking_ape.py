from typing import Tuple
from matplotlib import scale
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

from evo.tools import file_interface, plot
from evo.tools.plot import PlotMode
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from evo.tools.settings import SETTINGS

from completion import get_completion_stats, CompletionStats
from utils import load_trajectory


def parse_args():
    parser = ArgumentParser(
        description="Determine absolute pose error for a trajectory and its groundtruth",
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
        help="Dataset groundtruth file",
        action="store_true",
    )
    return parser.parse_args()


def get_sanitized_trajectories(
    tracking_csv: Path, groundtruth_csv: Path
) -> Tuple[PoseTrajectory3D, PoseTrajectory3D, CompletionStats]:
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
    # 2. In the second trajectory only the poses with closest timestamps to the first trajectory are kept.
    # A way of syncing trajectories a tad more meaningful for VR would be to
    # always use the previously tracked pose for each groundtruth pose.
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    return traj_est, traj_ref, c


def get_tracking_stats(
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
        import matplotlib.pyplot as plt

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
    tracking_csv = args.tracking_csv
    groundtruth_csv = args.groundtruth_csv
    show_plot = args.plot
    result, completion = get_tracking_stats(
        tracking_csv, groundtruth_csv, show_plot=show_plot
    )
    print(result)
    print(completion)

    # TODO: Try to reuse EVO settings as much as possible
    # TODO: Bring position/rpy graphs as well like in evo_traj


if __name__ == "__main__":
    main()
