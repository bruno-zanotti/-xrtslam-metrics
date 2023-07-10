#!/usr/bin/env python

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, cast

import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
from evo.core import lie_algebra as lie
from evo.core import sync
from evo.core.metrics import PoseRelation, Unit
from evo.core.result import Result
from evo.core.trajectory import PoseTrajectory3D
from evo.core.transformations import quaternion_from_matrix
from evo.tools import file_interface, plot
from evo.tools.plot import PlotMode
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mplcursors import HoverMode

from completion import CompletionStats
from utils import (
    COMPLETION_FULL_SINCE,
    DEFAULT_SEGMENT_DRIFT_TOLERANCE_M,
    SE3,
    SO3,
    Indices,
    Quaternion,
    error,
    make_color_iterator,
    make_dark_color_iterator,
    warn,
)


class TrackingPlot:
    def plot_reference_trajectory(
        self, traj_ref: PoseTrajectory3D, ref_name: str = "reference"
    ):
        raise NotImplementedError

    def plot_estimate_trajectory(self, *args, **kwargs):
        raise NotImplementedError

    def show(self):
        raise NotImplementedError


class TrajectoryErrorPlot(TrackingPlot):
    show_plot: bool = False
    plot_mode: PlotMode = PlotMode.xyz
    use_color_map: bool = False
    metric: str  # ate or rte
    ax: plt.Axes
    traj_ref: PoseTrajectory3D

    def __init__(
        self, show_plot: bool, plot_mode: str, use_color_map: bool, metric: str
    ) -> None:
        self.show_plot = show_plot
        if not self.show_plot:
            return

        self.plot_mode = PlotMode(plot_mode)
        self.use_color_map = use_color_map
        self.metric = metric
        self.fig = plt.figure()
        self.ax = plot.prepare_axis(self.fig, self.plot_mode)
        self.ax.set_title(f"Tracking error {metric.upper()} [m]")
        self.colors = make_color_iterator()

    def plot_reference_trajectory(
        self, traj_ref: PoseTrajectory3D, ref_name: str = "reference"
    ):
        if not self.show_plot:
            return
        self.traj_ref = traj_ref
        plot.traj(
            self.ax, self.plot_mode, traj_ref, style="--", color="gray", label=ref_name
        )

    def plot_estimate_trajectory(  # pylint: disable=arguments-differ
        self,
        traj_est: PoseTrajectory3D,
        result: Result,
        est_name: str = "estimate",
    ):
        if not self.show_plot:
            return

        errors = result.np_arrays["error_array"]
        min_error = result.stats["min"]
        max_error = result.stats["max"]

        if self.use_color_map:
            plot.traj_colormap(
                self.ax,
                traj_est,
                errors,
                self.plot_mode,
                min_map=min_error,
                max_map=max_error,
            )
        else:
            plot.traj(
                self.ax,
                self.plot_mode,
                traj_est,
                color=next(self.colors),
                label=est_name,
                alpha=0.75,
            )

    def show(self):
        plt.show()


class SegmentDriftErrorPlot(TrackingPlot):
    show_plot: bool = False
    plot_mode: PlotMode = PlotMode.xyz
    use_color_map: bool = False
    segment_color_map: bool = False
    metric: str  # should be seg
    ax: plt.Axes
    traj_ref: PoseTrajectory3D
    plotted_estimates: int = 0

    def __init__(
        self,
        show_plot: bool,
        plot_mode: str,
        use_color_map: bool,
        segment_color_map: bool,
        metric: str,
    ) -> None:
        self.show_plot = show_plot
        if not self.show_plot:
            return

        self.plot_mode = PlotMode(plot_mode)
        self.use_color_map = use_color_map
        self.segment_color_map = segment_color_map
        self.metric = metric
        self.fig = plt.figure()
        self.ax = plot.prepare_axis(self.fig, self.plot_mode)
        self.ax.set_title(f"Tracking error {metric.upper()} [m]")
        self.traj_plots = []
        self.error_plots = []

    def plot_reference_trajectory(
        self, traj_ref: PoseTrajectory3D, ref_name: str = "reference"
    ):
        if not self.show_plot:
            return
        self.traj_ref = traj_ref

        plot.traj(
            self.ax,
            self.plot_mode,
            traj_ref,
            style=".-",
            color="silver",
            label=ref_name,
        )
        # NOTE: Toggle these comments to enable hover cursor on groundtruth line
        # self.ax.lines[-1].timestamps = traj_ref.timestamps - traj_ref.timestamps[0]
        # self.traj_plots.append(self.ax.lines[-1])

    def plot_estimate_trajectory(  # pylint: disable=arguments-differ
        self,
        traj_est: PoseTrajectory3D,
        segments: List[PoseTrajectory3D],
        error_tolerance_per_segment: float,
        result: Result,
        ijk: Indices,
        est_name: str = "estimate",  # pylint: disable=unused-argument
    ):
        if not self.show_plot:
            return

        # plot.traj(
        #     self.ax,
        #     self.plot_mode,
        #     traj_est,
        #     style="o-",
        #     color="grey",
        #     label=est_name,
        # )
        # self.ax.lines[-1].timestamps = traj_est.timestamps - traj_est.timestamps[0]
        # self.traj_plots.append(self.ax.lines[-1])

        errors = result.np_arrays["errors"]
        error_points_est = result.np_arrays["error_points_est"]
        error_points_ref = result.np_arrays["error_points_ref"]

        # Plot segments
        if self.use_color_map:
            merged = merge_segments(segments)
            segment_errors = result.np_arrays["segment_errors"]

            plot.traj_colormap(
                self.ax,
                merged,
                errors,
                self.plot_mode,
                min_map=min(segment_errors),
                max_map=max(segment_errors),
            )
        elif self.segment_color_map:
            merged = merge_segments(segments)
            segment_errors = result.np_arrays["segment_errors"]

            merged_errors = np.zeros(len(errors))
            m, e = 0, 0
            while m < len(merged_errors):
                merged_errors[m] = segment_errors[e]
                if errors[m] > error_tolerance_per_segment:
                    e += 1
                m += 1

            plot.traj_colormap(
                self.ax,
                merged,
                merged_errors,
                self.plot_mode,
                min_map=min(segment_errors),
                max_map=max(segment_errors),
            )
        else:
            colors = (
                make_color_iterator()
                if self.plotted_estimates % 2 == 0
                else make_dark_color_iterator()
            )
            for i, segment in enumerate(segments):
                plot.traj(
                    self.ax, self.plot_mode, segment, color=next(colors), style=".-"
                )
                self.ax.lines[-1].timestamps = (
                    segment.timestamps[0] - traj_est.timestamps[0]
                ) + (segment.timestamps - segment.timestamps[0])
                self.ax.lines[-1].offset_frame = next(
                    i
                    for i, t in enumerate(traj_est.timestamps)
                    if t == segment.timestamps[0]
                )
                ps = segment.positions_xyz
                diff = ps[1:] - ps[:-1]  # type: ignore
                self.ax.lines[-1].lengths = np.hstack(
                    (np.linalg.norm(diff[:, ijk], axis=1), [0])
                )
                self.traj_plots.append(self.ax.lines[-1])

        # Plot red error lines when between segment ends and starts
        plot_ijk = [{"x": 0, "y": 1, "z": 2}[i] for i in self.plot_mode.value]
        errpoints_est = error_points_est[:, plot_ijk]
        errpoints_ref = error_points_ref[:, plot_ijk]
        error_lines = cast(Sequence, np.stack((errpoints_est, errpoints_ref), axis=1))
        dim = len(plot_ijk)
        if dim == 3:
            lines = Line3DCollection(error_lines, linestyles="--", colors="red")
        elif dim == 2:
            lines = LineCollection(error_lines, linestyles="--", colors="red")
        else:
            raise ValueError(f"Unexpected {dim=} {ijk=}")
        self.ax.add_collection(lines)  # type: ignore
        self.ax.collections[-1].errors = np.linalg.norm(
            error_points_ref - error_points_est, axis=1
        )
        self.error_plots.append(self.ax.collections[-1])
        self.ax.plot(*errpoints_est.T, ".", fillstyle="none", color="black")
        self.ax.plot(*errpoints_ref.T, ".", fillstyle="none", color="black")
        self.plotted_estimates += 1

    def show(self):
        if not self.show_plot:
            return

        # Set hover tooltips

        def on_hover_trajectory(s):
            i = int(s.index)
            time = s.artist.timestamps[i]
            text = f"{time:.2f}s"
            if hasattr(s.artist, "lengths"):
                text += f"\n{s.artist.lengths[i]:.4f}m"
            if hasattr(s.artist, "offset_frame"):
                text += f"\nframe {s.artist.offset_frame + i}"
            s.annotation.set_text(text)

        cursor = mplcursors.cursor(self.traj_plots, hover=HoverMode.Transient)
        cursor.connect("add", on_hover_trajectory)

        def on_hover_error_lines(s):
            s.annotation.set_text(f"{s.artist.errors[int(s.index[0])]:.4f}m")

        c = mplcursors.cursor(self.error_plots, hover=HoverMode.Transient)
        c.connect("add", on_hover_error_lines)
        plt.show()


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
    parser.set_defaults(use_color_map=None)
    parser.add_argument(
        "--color_map",
        "-cm",
        dest="use_color_map",
        action="store_true",
        help="Use color map for trajectory color based on error from groundtruth",
    )
    parser.add_argument(
        "--no_color_map",
        "-nocm",
        dest="use_color_map",
        action="store_false",
        help="Do not use color map for trajectory color based on error from groundtruth",
    )
    parser.add_argument(
        "--segment_color_map",
        "-scm",
        dest="segment_color_map",
        action="store_true",
        help="Use color map to paint entire segments in the segment-drift plot",
    )
    parser.add_argument(
        "--sd_tolerance",
        "-sdtol",
        type=float,
        default=DEFAULT_SEGMENT_DRIFT_TOLERANCE_M,
        help="Segment error tolerance for the SD metric",
    )
    parser.add_argument(
        "--sd_error_components",
        "-sdec",
        default="xyz",
        choices=["xy", "xz", "yx", "yz", "zx", "zy", "xyz"],
        help="Which axes to use for error computation in the SD metric",
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
    tracking_plot: TrajectoryErrorPlot | SegmentDriftErrorPlot,
    pose_relation: PoseRelation = PoseRelation.translation_part,
    alignment: int = 0,  # -1: origin, 0: umemaya, >0 align first n points
    silence: bool = False,
    sd_tolerance: float = DEFAULT_SEGMENT_DRIFT_TOLERANCE_M,
    sd_error_components: Sequence[int] = (0, 1, 2),
) -> Result:
    traj_est, traj_ref = get_sanitized_trajectories(
        tracking_csv, groundtruth_csv, silence=silence
    )
    est_name = str(tracking_csv)
    ref_name = "groundtruth"

    result = Result()

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
            ref_name=ref_name,
            est_name=est_name,
        )

        cast(TrajectoryErrorPlot, tracking_plot).plot_estimate_trajectory(
            traj_est,
            result,
            est_name=est_name,
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
            ref_name=ref_name,
            est_name=est_name,
            support_loop=False,  # Seems to only be used to not modify the input trajectories in jupyter notebooks
        )

        cast(TrajectoryErrorPlot, tracking_plot).plot_estimate_trajectory(
            traj_est,
            result,
            est_name=est_name,
        )
    elif metric == "seg":  # Segment Drift
        error_tolerance_per_segment = sd_tolerance
        ijk = np.array(sd_error_components, dtype=int)

        segments = []

        poses_count = len(traj_est.timestamps)

        i, ri = 0, 0
        traj_est.align_origin(traj_ref)
        remainder = deepcopy(traj_est)

        errors_list = []
        error_points_est_list = []
        error_points_ref_list = []
        while i < poses_count:
            p, e = get_point_error(remainder, traj_ref, ri, i, ijk)
            errors_list.append(e)
            if e > error_tolerance_per_segment:
                error_points_est_list.append(remainder.positions_xyz[ri, ijk])
                error_points_ref_list.append(p)
                segment, remainder = split_segment(traj_ref, traj_est, remainder, i, ri)
                segments.append(segment)
                errors_list.append(0)
                ri = 0
            i += 1
            ri += 1

        segment, remainder = split_segment(traj_ref, traj_est, remainder, i - 1, ri)
        if len(segment.timestamps) > 1:
            segments.append(segment)
        ri = 0

        errors = np.array(errors_list)
        error_points_est = np.array(error_points_est_list)
        error_points_ref = np.array(error_points_ref_list)

        seg_result = Result()
        seg_result.add_info(
            {
                "title": "Segment Drift",
                "ref_name": ref_name,
                "est_name": est_name,
                "label": "Segment Drift (m)",
            }
        )

        get_duration = lambda s: s.timestamps[-1] - s.timestamps[0]
        get_length = lambda s: sum(
            cast(float, np.linalg.norm((b - a)[ijk]))
            for a, b in zip(s.positions_xyz, s.positions_xyz[1:])
        )

        segments_count = len(segments)
        dataset_duration = get_duration(traj_est)
        dataset_length = get_length(traj_est)
        durations = np.array([get_duration(s) for s in segments])
        lengths = np.array([get_length(s) for s in segments])
        segment_errors = [e for e in errors_list if e >= error_tolerance_per_segment]
        segment_errors = np.append(segment_errors, error_tolerance_per_segment)
        assert len(segment_errors) == len(segments)
        seg_result.add_stats(
            {
                "Number of segments": segments_count,
                "Min error": min(e for e in errors if e != 0),  # Not very useful
                "Max error": max(errors),
                "Segments p/second": segments_count / dataset_duration,
                "Min segment duration": min(durations),
                "Max segment duration": max(durations),
                "Mean segment duration": np.mean(durations),
                "Median segment duration": np.median(durations),
                "Std segment duration": np.std(durations),
                # "Mean segment duration %": round(np.mean(durations) / dataset_duration * 100, 2),
                # "Median segment duration %": round(np.median(durations) / dataset_duration * 100, 2),
                # "Std segment duration %": round(np.std(durations) / dataset_duration * 100, 2),
                "Segments p/meter": segments_count / dataset_length,
                "Min segment length": min(lengths),
                "Max segment length": max(lengths),
                "Mean segment length": np.mean(lengths),
                "Median segment length": np.median(lengths),
                "Std segment length": np.std(lengths),
                # "Mean segment length %": round(np.mean(lengths) / dataset_length * 100, 2),
                # "Median segment length %": round(np.median(lengths) / dataset_length * 100, 2),
                # "Std segment length %": round(np.std(lengths) / dataset_length * 100, 2),
                "SDS": np.mean(segment_errors / durations),
                "SDS std": np.std(segment_errors / durations),
                "SDM": np.mean(segment_errors / lengths),
                "SDM std": np.std(segment_errors / lengths),
            }
        )

        seg_result.add_np_array("errors", errors)
        seg_result.add_np_array("segment_errors", segment_errors)
        seg_result.add_np_array("error_points_est", error_points_est)
        seg_result.add_np_array("error_points_ref", error_points_ref)

        seg_result.add_trajectory(ref_name, traj_ref)
        seg_result.add_trajectory(est_name, traj_est)
        seg_result.add_np_array("timestamps", traj_est.timestamps)
        seg_result.add_np_array("distances_from_start", traj_ref.distances)
        seg_result.add_np_array("distances", traj_est.distances)
        result = seg_result
        print(f"finished {tracking_csv} {' ' * 80}")

        cast(SegmentDriftErrorPlot, tracking_plot).plot_estimate_trajectory(
            traj_est,
            segments,
            error_tolerance_per_segment,
            seg_result,
            ijk=ijk,
            est_name=est_name,
        )

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

    # Print
    i_ts = traj_est.timestamps[i]
    i0_ts = remainder.timestamps[0]
    i0 = next(i for i, t in enumerate(traj_est.timestamps) if t == i0_ts)
    total_s = traj_est.timestamps[-1] - traj_est.timestamps[0]
    current_s = i_ts - traj_est.timestamps[0]
    print(
        f"Segment {100*current_s/total_s:.2f}% {current_s:.0f}/{total_s:.0f}s "
        f"id=[{i0}, {i}] ts=[{i0_ts}, {i_ts}]",
        end="\r",
    )

    # Split
    segment = create_subtrajectory(remainder, 0, ri + 1)
    remainder.reduce_to_time_range(i_ts, end_timestamp=None)
    align_origin_at(remainder, traj_ref, i)

    return segment, remainder


def create_subtrajectory(traj: PoseTrajectory3D, i: int, j: int) -> PoseTrajectory3D:
    stamps = traj.timestamps[i:j]
    poses = traj.poses_se3[i:j]
    return PoseTrajectory3D(timestamps=stamps, poses_se3=poses)


def transform_trajectory(traj: PoseTrajectory3D, t: SE3):
    """Same as PoseTrajectory3D.transform but faster due to skipping some
    cache generation"""

    traj._poses_se3 = [  # pylint: disable=protected-access
        np.dot(t, p) for p in traj.poses_se3
    ]

    # Invalidate cache
    if hasattr(traj, "_positions_xyz"):
        del traj._positions_xyz
    if hasattr(traj, "_orientations_quat_wxyz"):
        del traj._orientations_quat_wxyz


def align_origin_at(a: PoseTrajectory3D, b: PoseTrajectory3D, i: int = 0) -> SE3:
    """
    align the origin to the origin of a reference trajectory
    :param a: trajectory to align
    :param b: reference trajectory
    :return: the used transformation
    """
    if a.num_poses == 0 or b.num_poses == 0:
        raise ValueError("can't align an empty trajectory...")
    traj_origin = a.poses_se3[0]
    traj_ref_origin = b.poses_se3[i]
    to_ref_origin = np.dot(traj_ref_origin, lie.se3_inverse(traj_origin))
    transform_trajectory(a, to_ref_origin)

    # After a couple of transforms the rotation lose determinant 1 due to
    # floating point errors so we re-orthonormalize every so often
    if abs(np.linalg.det(to_ref_origin[:3, :3]) - 1) > 1e-9:
        orthonormalize_rotations(a)
    return to_ref_origin


def matrix_from_quaternion(q: Quaternion) -> SO3:
    """Return a rotation matrix from a quaternion."""
    qw, qx, qy, qz = q
    r0 = [2 * (qw * qw + qx * qx) - 1, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)]
    r1 = [2 * (qx * qy + qw * qz), 2 * (qw * qw + qy * qy) - 1, 2 * (qy * qz - qw * qx)]
    r2 = [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 2 * (qw * qw + qz * qz) - 1]
    return np.array([r0, r1, r2])


def orthonormalize_rotations(traj: PoseTrajectory3D):
    for se3 in traj.poses_se3:
        q = quaternion_from_matrix(se3)
        q = q / np.linalg.norm(q)
        # se3[:] = quaternion_matrix(q) # For some reason this didn't work
        se3[:3, :3] = matrix_from_quaternion(q)


def get_point_error(
    a: PoseTrajectory3D,
    b: PoseTrajectory3D,
    ai: int,
    bi: int,
    ijk: Indices = np.array([0, 1, 2]),
):
    pa = a.positions_xyz[ai][ijk]
    pb = b.positions_xyz[bi][ijk]
    ta = a.timestamps[ai]
    tb = b.timestamps[bi]
    if tb != ta:
        ps, ts = b.positions_xyz, b.timestamps
        l = bi if tb < ta else bi - 1
        r = l + 1
        tl = ts[l] if l >= 0 else ts[0] - (ts[1] - ts[0])
        pl = (ps[l] if l >= 0 else ps[0] - (ps[1] - ps[0]))[ijk]
        tr = ts[r] if r < len(ts) else ts[-1] + (ts[-1] - ts[-2])
        pr = (ps[r] if r < len(ps) else ps[-1] + (ps[-1] - ps[-2]))[ijk]
        assert tl <= ta and ta <= tr
        pb = pl + (pr - pl) * ((ta - tl) / (tr - tl))
    e = np.linalg.norm(pa - pb)
    return pb, e


def get_tracking_stats(
    metric: str,  # rte, ate, seg
    tracking_csvs: List[Path],
    groundtruth_csv: Path,
    pose_relation: PoseRelation = PoseRelation.translation_part,
    alignment: int = 0,  # -1: origin, 0: umemaya, >0 align first n points
    show_plot: bool = False,
    plot_mode: str = "xyz",  # "xz", "xy", etc
    use_color_map: bool = False,
    segment_color_map: bool = False,
    silence: bool = False,
    sd_tolerance: float = DEFAULT_SEGMENT_DRIFT_TOLERANCE_M,
    sd_error_components: Sequence[int] = (0, 1, 2),
) -> Dict[Path, Result]:
    tracking_plot = (
        SegmentDriftErrorPlot(
            show_plot, plot_mode, use_color_map, segment_color_map, metric
        )
        if metric == "seg"
        else TrajectoryErrorPlot(show_plot, plot_mode, use_color_map, metric)
    )
    _, gt = get_sanitized_trajectories(  # NOTE: sanitizing only against first traj
        tracking_csvs[0], groundtruth_csv, silence=True
    )
    tracking_plot.plot_reference_trajectory(gt)
    results = {}
    for tracking_csv in tracking_csvs:
        result = compute_tracking_stats(
            metric,
            tracking_csv,
            groundtruth_csv,
            tracking_plot,
            pose_relation=pose_relation,
            alignment=alignment,
            silence=silence,
            sd_tolerance=sd_tolerance,
            sd_error_components=sd_error_components,
        )
        results[tracking_csv] = result
        print(f"File {tracking_csv}")
        print(result)

    tracking_plot.show()
    return results


def main():
    args = parse_args()
    metric = args.metric
    groundtruth_csv = args.groundtruth_csv
    tracking_csvs = args.tracking_csvs
    show_plot = args.plot
    plot_mode = args.plot_mode
    use_color_map = args.use_color_map
    segment_color_map = args.segment_color_map
    sd_tolerance = args.sd_tolerance
    sd_error_components = args.sd_error_components

    sd_error_components = [{"x": 0, "y": 1, "z": 2}[i] for i in sd_error_components]
    if use_color_map is None:
        use_color_map = metric in ["ate", "rte"] and len(tracking_csvs) == 1

    get_tracking_stats(
        metric,
        tracking_csvs,
        groundtruth_csv,
        show_plot=show_plot,
        plot_mode=plot_mode,
        use_color_map=use_color_map,
        segment_color_map=segment_color_map,
        sd_tolerance=sd_tolerance,
        sd_error_components=sd_error_components,
    )

    # TODO: Try to reuse EVO settings as much as possible
    # TODO: Bring position/rpy graphs as well like in evo_traj


if __name__ == "__main__":
    main()
