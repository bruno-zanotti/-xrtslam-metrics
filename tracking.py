#!/usr/bin/env python

from typing import Dict, List, Tuple, Sequence
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import mplcursors
import matplotlib.pyplot as plt
from copy import deepcopy
from evo.tools.settings import SETTINGS
from evo.tools import file_interface, plot
from evo.tools.plot import PlotMode
from evo.core import sync
import evo.core.geometry as geometry
from evo.core import lie_algebra as lie
from evo.core.result import Result
import evo.core.transformations as tr
from evo.core.trajectory import PoseTrajectory3D, PosePath3D, merge
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation, Unit
import logging
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from completion import CompletionStats
from utils import COMPLETION_FULL_SINCE, error, make_color_iterator, warn

logger = logging.getLogger(__name__)  # TODO: what's this for?


def parse_args():
    parser = ArgumentParser(
        description="Determine absolute pose error for a trajectory and its groundtruth",
    )
    parser.add_argument(
        "metric",
        type=str,
        help="What tracking metric to compute",
        default="ate",
        choices=["ate", "rte", "seg"],
    )
    parser.add_argument(
        "groundtruth_csv",
        type=Path,
        help="Dataset groundtruth file",
    )
    parser.add_argument(
        "tracking_csvs",
        type=Path,
        nargs="+",
        help="Tracking files generated from Monado to compare",
    )
    parser.add_argument(
        "-p",
        "--plot",
        help="Enable to show trajectory plot",
        action="store_true",
    )
    parser.add_argument(
        "--plot_mode",
        "-pm",
        default="xyz",
        help="Axes of the trajectory to plot",
        choices=["xy", "xz", "yx", "yz", "zx", "zy", "xyz"],
    )
    return parser.parse_args()


def get_sanitized_trajectories(
    tracking_csv: Path, groundtruth_csv: Path, silence=False
) -> Tuple[PoseTrajectory3D, PoseTrajectory3D]:
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
    # TODO: Make it work with the entire trajectory, problem in alignment as of now
    first_ts = traj_ref.timestamps[16]  # 16 - 8890 is bad, 17-8890 is good
    last_ts = traj_ref.timestamps[8890]  # Appears idalamer
    traj_ref.reduce_to_time_range(first_ts, last_ts)
    traj_est.reduce_to_time_range(first_ts, last_ts)

    c = CompletionStats(
        e0, e1, r0, r1, traj_est.timestamps.size, traj_ref.timestamps.size
    )

    if c.tracking_completion < COMPLETION_FULL_SINCE and not silence:
        warn(
            f"Tracking completion for {tracking_csv} is "
            f"{c.tracking_completion * 100:.2f}% < {COMPLETION_FULL_SINCE * 100:.2f}%",
            "Tracking metrics will be unreliable.",
        )

    # TODO: PR with a more realtime-appropriate trajectory alignment.
    # `associate_trajectories`` synchronizes the two trajectories as follows:
    # 1. The trajectory with less poses is kept
    # 2. In the second trajectory only the poses with closest timestamps to the
    #    first trajectory are kept.
    # A way of syncing trajectories a tad more meaningful for VR would be to
    # always use the previously tracked pose for each groundtruth pose.
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    return traj_est, traj_ref


def compute_tracking_stats(
    metric: str,  # rte, ate
    tracking_csv: Path,
    groundtruth_csv: Path,
    pose_relation: PoseRelation = PoseRelation.translation_part,
    alignment: int = 0,  # -1: origin, 0: umemaya, >0 align first n points
    silence: bool = False,
    plot_mode_str: str = "xyz",
) -> Result:
    traj_est, traj_ref = get_sanitized_trajectories(
        tracking_csv, groundtruth_csv, silence=silence
    )
    if metric == "ate":
        # NOTE: Possible issues for VR.
        # - Umemaya alignment does not account how off are we from the starting point
        # - Only considering translation error, maybe rotational part is important too?
        result = main_ape.ape(
            traj_ref=traj_ref,
            traj_est=traj_est,
            pose_relation=pose_relation,
            align=alignment >= 0,
            correct_scale=False,
            n_to_align=alignment if alignment > 0 else -1,
            align_origin=alignment == -1,
            ref_name="groundtruth",
            est_name=tracking_csv,
        )
    elif metric == "rte":
        # NOTE: Possible issues for VR.
        # - Umemaya alignment seems to certainly be a bad idea for relative error, align_first_n sounds better
        # - Here again, only translation error considered, maybe rotational part is very important?
        result = main_rpe.rpe(
            traj_ref=traj_ref,
            traj_est=traj_est,
            pose_relation=pose_relation,
            delta=6,
            delta_unit=Unit.frames,  # TODO: Evo doesn't support delta_units=seconds, I think 0.2s would be good, delta=6 is an approximation to that
            rel_delta_tol=0.1,  # only used when all_pairs is enabled
            all_pairs=False,  # TODO: use all_pairs?
            align=alignment >= 0,
            correct_scale=False,
            n_to_align=alignment if alignment > 0 else -1,
            align_origin=alignment == -1,
            ref_name="groundtruth",
            est_name=tracking_csv,
            support_loop=False,  # Seems to only be used to not modify the input trajectories in jupyter notebooks
        )
    elif metric == "seg":
        dim = len(plot_mode_str)

        INITIAL_ALIGNMENT_TIME_S = 5
        SEGMENT_ALIGNMENT_TIME_S = 1
        ERROR_TOLERANCE_PER_SEGMENT_M = 0.01

        segments = []

        poses_count = len(traj_est.timestamps)

        i, ri = 0, 0
        traj_est.align_origin(traj_ref)
        remainder = deepcopy(traj_est)

        # errors = np.zeros(poses_count)
        errors = []
        error_points_est = []
        error_points_ref = []
        while i < poses_count:
            p, e = get_point_error(remainder, traj_ref, ri, i, dim)
            errors.append(e)
            if e > ERROR_TOLERANCE_PER_SEGMENT_M:
                error_points_est.append(remainder.positions_xyz[ri][0:dim])
                error_points_ref.append(p)
                segment, remainder = split_segment(traj_ref, traj_est, remainder, i, ri)
                segments.append(segment)
                errors.append(0)
                ri = 0
            i += 1
            ri += 1

        segment, remainder = split_segment(traj_ref, traj_est, remainder, i - 1, ri)
        segments.append(segment)
        ri = 0

        error_points_est = np.array(error_points_est)
        error_points_ref = np.array(error_points_ref)

        seg_result = Result()
        metric_name = "Segments"
        seg_result.add_info(
            {
                "title": "Segments Metric TITLE",
                "ref_name": "groundtruth",
                "est_name": tracking_csv,
                "label": "{} {}".format(metric_name, "({})".format("m METERS")),
            }
        )
        seg_result.add_stats(
            {
                # TODO: Report number of segments, and median/std/etc of length
                # of segments
                # "rmse"
                # "mean"
                # "median"
                # "std"
                # "min": min(error_array),
                # "max": max(error_array),
                "min": 0,
                "max": ERROR_TOLERANCE_PER_SEGMENT_M,
                # "sse"
            }
        )

        # TODO: Review these results, specially the names I gave
        # TODO: Review comments and remove dangling ones
        seg_result.add_np_array("errors", errors)
        seg_result.add_np_array("error_points_est", error_points_est)
        seg_result.add_np_array("error_points_ref", error_points_ref)

        seg_result.info["title"] = "Segment Metric TITTLTLTLE"
        logger.info(seg_result.pretty_str())
        seg_result.add_trajectory("groundtruth", traj_ref)
        seg_result.add_trajectory(tracking_csv, traj_est)
        seg_result.add_np_array("timestamps", traj_est.timestamps)
        seg_result.add_np_array("distances_from_start", traj_ref.distances)
        seg_result.add_np_array("distances", traj_est.distances)

        # TODO: Move plotting outside of this if possible to match behaviour of
        # other tracking metrics
        fig, plot_mode = plt.figure(), PlotMode(plot_mode_str)
        ax = plot.prepare_axis(fig, plot_mode)
        plots = []
        seconds_from_start = traj_ref.timestamps - traj_ref.timestamps[0]

        plot.traj(ax, plot_mode, traj_ref, style="o-", color="gray", label="ref")
        ax.lines[-1].timestamps = seconds_from_start
        plots.append(ax.lines[-1])

        plot.traj(ax, plot_mode, traj_est, style="o-", color="black", alpha=0.1)
        ax.lines[-1].timestamps = seconds_from_start
        plots.append(ax.lines[-1])

        colors = make_color_iterator()
        COLORMAP = False
        if COLORMAP:
            merged = merge_segments(segments)
            maxerr = ERROR_TOLERANCE_PER_SEGMENT_M
            plot.traj_colormap(ax, merged, errors, plot_mode, min_map=0, max_map=maxerr)
        else:
            for i, segment in enumerate(segments):
                plot.traj(ax, plot_mode, segment, color=next(colors), style="o-")
                ax.lines[-1].timestamps = seconds_from_start
                plots.append(ax.lines[-1])

        error_lines = np.stack((error_points_est, error_points_ref), axis=1)
        if dim == 3:
            lines = Line3DCollection(error_lines, linestyles="--", colors="red")
        elif dim == 2:
            lines = LineCollection(error_lines, linestyles="--", colors="red")
        else:
            raise Exception(f"Unexpected {dim=}")
        ax.add_collection(lines)

        ax.plot(*error_points_est[:, 0:dim].T, ".", color="black")
        ax.plot(*error_points_ref[:, 0:dim].T, ".", color="black")

        # Set hover tooltip
        cursor = mplcursors.cursor(plots, hover=mplcursors.HoverMode.Transient)
        cursor.connect(
            "add",
            lambda s: s.annotation.set_text(f"{s.artist.timestamps[int(s.index)]:.2f}"),
        )

        plt.show()

        return seg_result
    else:
        error("Unexpected branch taken")

    return result


def merge_segments(trajectories: Sequence[PoseTrajectory3D]) -> PoseTrajectory3D:
    merged_stamps = np.concatenate([t.timestamps for t in trajectories])
    merged_xyz = np.concatenate([t.positions_xyz for t in trajectories])
    merged_quat = np.concatenate([t.orientations_quat_wxyz for t in trajectories])
    return PoseTrajectory3D(merged_xyz, merged_quat, merged_stamps)


def split_segment(
    traj_ref: PoseTrajectory3D,
    traj_est: PoseTrajectory3D,
    remainder: PoseTrajectory3D,
    i: int,
    ri: int,
):
    """

    Cuts the trajectory `remainder` of traj_est at index ri (index i w.r.t
    traj_est, ri w.r.t. remainder), it returns a pair with:
    - first element being a segment from start of `remainder` up to index ri unmodified
    - second element being the remainder of the trajectory from index i, but aligned with
    traj_ref by matching point i with traj_ref[i]

    NOTE: This function modifies `remainder`.
    """

    segment = create_subtrajectory(remainder, 0, ri + 1)
    i_ts = traj_est.timestamps[i]
    print(
        f"{i=} {i_ts=} from_start={i_ts - traj_est.timestamps[0]:.2f} max={traj_est.timestamps[-1] - traj_est.timestamps[0]:.2f}"
    )
    remainder.reduce_to_time_range(i_ts, end_timestamp=None)
    align_origin_at(remainder, traj_ref, i)

    return segment, remainder


def create_subtrajectory(traj: PoseTrajectory3D, i: int, j: int) -> PoseTrajectory3D:
    stamps = traj.timestamps[i:j]
    xyz = traj.positions_xyz[i:j]
    quat = traj.orientations_quat_wxyz[i:j]
    return PoseTrajectory3D(xyz, quat, stamps)


def align_origin_at(a: PoseTrajectory3D, b: PoseTrajectory3D, i: int = 0) -> np.ndarray:
    """
    align the origin to the origin of a reference trajectory
    :param a: trajectory to align
    :param b: reference trajectory
    :return: the used transformation
    """
    if a.num_poses == 0 or b.num_poses == 0:
        raise Exception("can't align an empty trajectory...")
    traj_origin = a.poses_se3[0]
    traj_ref_origin = b.poses_se3[i]
    to_ref_origin = np.dot(traj_ref_origin, lie.se3_inverse(traj_origin))
    logger.debug("Origin alignment transformation:\n{}".format(to_ref_origin))
    a.transform(to_ref_origin)
    return to_ref_origin


def get_point_error(
    a: PoseTrajectory3D, b: PoseTrajectory3D, ai: int, bi: int, dim: int = 3
):
    pa = a.positions_xyz[:, 0:dim][ai]
    pb = b.positions_xyz[:, 0:dim][bi]
    ta = a.timestamps[ai]
    tb = b.timestamps[bi]
    if tb != ta:
        ps, ts = b.positions_xyz[:, 0:dim], b.timestamps
        l = bi - 1 if tb > ta else bi
        r = bi + 1 if tb < ta else bi
        pb = ps[l] + (ps[r] - ps[l]) * ((ta - ts[l]) / (ts[r] - ts[l]))
    e = np.linalg.norm(pa - pb)
    return pb, e


def se3_poses_to_xyz_quat_wxyz(
    poses: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    xyz = np.array([pose[:3, 3] for pose in poses])
    quat_wxyz = np.array([tr.quaternion_from_matrix(pose) for pose in poses])
    return xyz, quat_wxyz


def transform_from(traj_est: PoseTrajectory3D, t: np.ndarray, i: int) -> None:
    traj_est._poses_se3[i:] = [np.dot(t, p) for p in traj_est.poses_se3[i:]]
    (
        traj_est._positions_xyz[i:],
        traj_est._orientations_quat_wxyz[i:],
    ) = se3_poses_to_xyz_quat_wxyz(traj_est.poses_se3[i:])


def get_tracking_stats(
    metric: str,  # rte, ate, seg
    tracking_csvs: List[Path],
    groundtruth_csv: Path,
    pose_relation: PoseRelation = PoseRelation.translation_part,
    alignment: int = 0,  # -1: origin, 0: umemaya, >0 align first n points
    show_plot: bool = False,
    plot_mode: str = "xyz",  # "xz", "xy", etc
    silence: bool = False,
) -> Dict[Path, Result]:
    results = {}
    ref_name = str(groundtruth_csv)
    for tracking_csv in tracking_csvs:
        result = compute_tracking_stats(
            metric,
            tracking_csv,
            groundtruth_csv,
            pose_relation,
            alignment,
            silence,
            plot_mode,
        )
        results[tracking_csv] = result

    if show_plot:
        fig = plt.figure()
        plot_mode = PlotMode(plot_mode)
        ax = plot.prepare_axis(fig, plot_mode)
        _, gt = get_sanitized_trajectories(  # NOTE: sanitizing only against first traj
            tracking_csvs[0], groundtruth_csv, silence=True
        )
        plot.traj(ax, plot_mode, gt, style="--", color="gray", label=groundtruth_csv)
        colors = make_color_iterator()

        for tracking_csv, result in results.items():
            # TODO: seg metric coesnt support more than one csv yet
            if len(tracking_csvs) == 1:
                # TODO: This doesnt work with seg yet
                plot.traj_colormap(
                    ax,
                    result.trajectories[tracking_csv],
                    result.np_arrays["errors"],
                    plot_mode,
                    min_map=result.stats["min"],
                    max_map=result.stats["max"],
                )
                # ax.plot(result.np_arrays['interest_points'])
            else:
                plot.traj(
                    ax,
                    plot_mode,
                    result.trajectories[tracking_csv],
                    color=next(colors),
                    label=tracking_csv,
                    alpha=0.75,
                )
        plt.title("Tracking error")
        plt.show()

    return results


def main():
    args = parse_args()
    metric = args.metric
    groundtruth_csv = args.groundtruth_csv
    tracking_csvs = args.tracking_csvs
    show_plot = args.plot
    plot_mode = args.plot_mode

    results = get_tracking_stats(
        metric, tracking_csvs, groundtruth_csv, show_plot=show_plot, plot_mode=plot_mode
    )

    for tracking_csv, result in results.items():
        print(f"File {tracking_csv}")
        print(result)

    # TODO: Try to reuse EVO settings as much as possible
    # TODO: Bring position/rpy graphs as well like in evo_traj


if __name__ == "__main__":
    main()
