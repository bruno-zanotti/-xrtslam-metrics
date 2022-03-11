from matplotlib import scale
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

from evo.tools import file_interface, plot
from evo.tools.plot import PlotMode
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from evo.tools.settings import SETTINGS

from completion import get_completion_stats
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


def main():
    args = parse_args()
    tracking_csv = args.tracking_csv
    groundtruth_csv = args.groundtruth_csv
    show_plot = args.plot

    # TODO: PR with a more realtime-appropriate trajectory alignment.
    # Evo synchronizes the two trajectories as follows:
    # 1. The trajectory with less poses is kept
    # 2. In the second trajectory only the poses with closest timestamps to the first trajectory are kept.
    # A way of syncing trajectories a tad more realistic for VR would be to
    # always use the previously tracked pose for each groundtruth pose.

    tracked = load_trajectory(tracking_csv)
    gt = load_trajectory(groundtruth_csv)

    # Read trajectorjes
    # NOTE: Evo uses doubles for its timestamps and thus looses a bit of
    # precision, but even in the worst case, the precision is about ~1usec
    traj_ref = file_interface.read_euroc_csv_trajectory(groundtruth_csv)
    traj_est = file_interface.read_euroc_csv_trajectory(tracking_csv)

    # Trim both trajectories so that only overlapping timestamps are kept
    c = get_completion_stats(tracked, gt)
    print(c)
    first_ts = max(c.first_tracked_ts, c.first_gt_ts)
    last_ts = min(c.last_tracked_ts, c.last_gt_ts)
    traj_ref.reduce_to_time_range(first_ts, last_ts)
    traj_est.reduce_to_time_range(first_ts, last_ts)
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    # TODO: I might want to also consider rotation_angle_deg
    pose_relation = PoseRelation.translation_part
    ref_name = "reference"
    est_name = "estimated"
    umemaya_align = True  # Otherwise align_origin
    result = main_ape.ape(
        traj_ref=traj_ref,
        traj_est=traj_est,
        pose_relation=pose_relation,
        align=umemaya_align,
        correct_scale=False,
        n_to_align=-1,
        align_origin=umemaya_align,  # TODO: Doesn't this make a lot of sense for VR?
        ref_name=ref_name,
        est_name=est_name,
    )
    print(result)

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
    print(result)

    # TODO: Report somehow when completion.percentage < 90%
    # TODO: Try to reuse EVO settings as much as possible
    # TODO: Bring position/rpy graphs as well like in evo_traj

if __name__ == "__main__":
    main()
