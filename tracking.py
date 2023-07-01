#!/usr/bin/env python

from typing import Dict, List, Tuple, Sequence
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from copy import deepcopy

import matplotlib.pyplot as plt
from evo.tools import file_interface, plot
from evo.tools.plot import PlotMode
from evo.core import sync
import evo.core.geometry as geometry
from evo.core import lie_algebra as lie
from evo.core.result import Result
import evo.core.transformations as tr
from evo.core.trajectory import PoseTrajectory3D, PosePath3D
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation, Unit
import logging

logger = logging.getLogger(__name__)

from completion import CompletionStats
from utils import COMPLETION_FULL_SINCE, error, make_color_iterator, warn


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

        # stamps = np.divide(mat[:, 0], 1e9)  # n x 1  -  nanoseconds to seconds
        # xyz = mat[:, 1:4]  # n x 3
        # quat = mat[:, 4:8]  # n x 4
        # logger.debug("Loaded {} stamps and poses from: {}".format(
        #     len(stamps), file_path))
        # return PoseTrajectory3D(xyz, quat, stamps)

        INITIAL_ALIGNMENT_TIME_S = 5
        SEGMENT_ALIGNMENT_TIME_S = 1
        ERROR_TOLERANCE_PER_SEGMENT_M = 0.120

        segments = []

        # seconds_from_start = np.array(
        #     [t - traj_est.timestamps[0] for t in traj_est.timestamps]
        # )
        poses_count = len(traj_est.timestamps)

        # initial_alignment_n = next(
        #     i for i, s in seconds_from_start if s >= INITIAL_ALIGNMENT_TIME_S
        # )
        i = 0
        traj_est.align_origin(traj_ref)
        remainder = deepcopy(traj_est)
        # segment, remainder = split_segment(traj_ref, traj_est, remainder, i)
        # segments.append(segment)

        # r_a, t_a, s = geometry.umeyama_alignment(
        #     traj_est.positions_xyz[0:initial_alignment_n, :].T,
        #     traj_ref.positions_xyz[0:initial_alignment_n, :].T,
        #     False,
        # )
        # transform_from(traj_est, lie.se3(r_a, t_a), 0)

        point_errors = np.zeros(poses_count)
        while i < poses_count:
            e = get_point_error(traj_est, traj_ref, i)
            point_errors[i] = e
            if e > ERROR_TOLERANCE_PER_SEGMENT_M:
                print(e)
                segment, remainder = split_segment(traj_ref, traj_est, remainder, i)
                segments.append(segment)
            i += 1

        # i = 0
        # interest_points[i] = traj_est.positions_xyz[i]

        # fig, plot_mode = plt.figure(), PlotMode("xy")
        # ax = plot.prepare_axis(fig, plot_mode)
        # plot.traj(ax, plot_mode, traj_ref, style="--", color="gray", label="gt")
        # plot.traj(ax, plot_mode, traj_est, label=f"{i=}", alpha=0.75)
        # plt.show()
        # print(f"{poses_count=}")
        # while i < poses_count:
        #     error3d = traj_est.positions_xyz[i] - traj_ref.positions_xyz[i]
        #     error1d = np.linalg.norm(error3d)
        #     error_array[i] = error1d
        #     print(f"{i=} {error1d=}")
        #     if error1d > ERROR_TOLERANCE_PER_SEGMENT_M:
        #         print("RESET")
        #         interest_points[i] = traj_est.positions_xyz[i]

        #         # Look for index that is 1 second after i
        #         forward_1s_n = i
        #         seconds_from_start_i = seconds_from_start[i]
        #         while (
        #             forward_1s_n < poses_count
        #             and seconds_from_start[forward_1s_n] < seconds_from_start_i + 1
        #         ):
        #             forward_1s_n += 1

        #         r_a, t_a, s = geometry.umeyama_alignment(
        #             traj_est.positions_xyz[i:forward_1s_n, :].T,
        #             traj_ref.positions_xyz[i:forward_1s_n, :].T,
        #             False,
        #         )

        #         transform_from(traj_est, lie.se3(r_a, t_a), i)

        #         fig, plot_mode = plt.figure(), PlotMode("xy")
        #         ax = plot.prepare_axis(fig, plot_mode)
        #         plot.traj(ax, plot_mode, traj_ref, style="--", color="gray", label="gt")
        #         plot.traj(ax, plot_mode, traj_est, label=f"{i=}", alpha=0.75)
        #         plt.show()
        #     i += 1

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
        seg_result.add_np_array("error_array", point_errors)

        seg_result.info["title"] = "Segment Metric TITTLTLTLE"
        logger.info(seg_result.pretty_str())
        seg_result.add_trajectory("groundtruth", traj_ref)
        seg_result.add_trajectory(tracking_csv, traj_est)
        # seg_result.add_np_array("seconds_from_start", seconds_from_start)
        seg_result.add_np_array("timestamps", traj_est.timestamps)
        seg_result.add_np_array("distances_from_start", traj_ref.distances)
        seg_result.add_np_array("distances", traj_est.distances)
        # seg_result.add_np_array("interest_points", interest_points)

        fig, plot_mode = plt.figure(), PlotMode("xy")
        ax = plot.prepare_axis(fig, plot_mode)
        plot.traj(ax, plot_mode, traj_ref, style="--", color="gray", label="gt")
        colors = make_color_iterator()
        for i, segment in enumerate(segments):
            plot.traj(
                ax, plot_mode, segment, label=f"{i=}", alpha=0.75, color=next(colors)
            )
            # plot.traj_colormap(
            #     ax,
            #     segment,
            #     point_errors,
            #     plot_mode,
            #     min_map=result.stats["min"],
            #     max_map=result.stats["max"],
            # )
        plt.show()

        return seg_result

        # result = {}
        # trajectories[tracking_csv],
        # np_arrays["error_array"]

        # align the trajectories
    else:
        error("Unexpected branch taken")
    return result


def split_segment(
    traj_ref: PoseTrajectory3D,
    traj_est: PoseTrajectory3D,
    remainder: PoseTrajectory3D,
    i: int,
):
    """

    Cuts the trajectory `remainder` of traj_est at index i (index i w.r.t traj_est), it returns a pair with
    - first element being a segment from start of `remainder` up to index i unmodified
    - second element being the remainder of the trajectory from index i, but
    aligned with traj_ref by matching point i with traj_ref[i]

    NOTE: This function modifies `remainder`.
    """

    # seconds_from_start = np.array(
    #     [t - traj_est.timestamps[0] for t in traj_est.timestamps]
    # )
    # initial_alignment_n = next(
    #     i for i, s in seconds_from_start if s >= INITIAL_ALIGNMENT_TIME_S
    # )
    # r_a, t_a, s = geometry.umeyama_alignment(
    #     traj_est.positions_xyz[0:initial_alignment_n, :].T,
    #     traj_ref.positions_xyz[0:initial_alignment_n, :].T,
    #     False,
    # )
    # transform_from(traj_est, lie.se3(r_a, t_a), 0)

    remainder_ts0 = remainder.timestamps[0]
    remainder_start_i = next(i for i, ts in enumerate(traj_est.timestamps) if ts == remainder_ts0)
    remainder_i = i - remainder_start_i
    segment = create_subtrajectory(remainder, 0, remainder_i + 1)
    print(f"{remainder_i=} {remainder_start_i=} {i=}")

    i_ts = traj_est.timestamps[i]
    remainder.reduce_to_time_range(i_ts, end_timestamp=None)
    align_origin_at_ref_idx(remainder, traj_ref, i)

    return segment, remainder


def create_subtrajectory(traj: PoseTrajectory3D, i: int, j: int) -> PoseTrajectory3D:
    stamps = traj.timestamps[i:j]
    xyz = traj.positions_xyz[i:j]
    quat = traj.orientations_quat_wxyz[i:j]
    return PoseTrajectory3D(xyz, quat, stamps)


def align_origin_at_ref_idx(
    traj: PoseTrajectory3D, traj_ref: PoseTrajectory3D, i: int = 0
) -> np.ndarray:
    """
    align the origin to the origin of a reference trajectory
    :param traj_ref: reference trajectory
    :return: the used transformation
    """
    if traj.num_poses == 0 or traj_ref.num_poses == 0:
        raise Exception("can't align an empty trajectory...")
    traj_origin = traj.poses_se3[0]
    traj_ref_origin = traj_ref.poses_se3[i]
    to_ref_origin = np.dot(traj_ref_origin, lie.se3_inverse(traj_origin))
    logger.debug("Origin alignment transformation:\n{}".format(to_ref_origin))
    traj.transform(to_ref_origin)
    return to_ref_origin


def get_point_error(full: PoseTrajectory3D, traj_ref: PoseTrajectory3D, i: int):
    error3d = full.positions_xyz[i] - traj_ref.positions_xyz[i]
    error = np.linalg.norm(error3d)
    return error


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
            metric, tracking_csv, groundtruth_csv, pose_relation, alignment, silence
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
            if len(tracking_csvs) == 1:
                plot.traj_colormap(
                    ax,
                    result.trajectories[tracking_csv],
                    result.np_arrays["error_array"],
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
