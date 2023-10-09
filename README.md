# Monado visual-inertial tracking measurement tools

This project contains scripts based on
[evo](https://github.com/MichaelGrupp/evo/wiki/Metrics) for automating the
evaluation of VIO/SLAM tracking systems in Monado.

See [this blogpost](https://mateosss.github.io/blog/xrtslam-metrics) for an
overview on what you can expect from this file and how to generate data to
analyze from Monado.

## Dependencies

Run `poetry install` on the root directory.

## Usage examples

### Timing

See a plot of the times each pose from the dataset took to compute. If no
start/end CSV column names are specified the script assumes defaults (see
`DEFAULT_TIMING_COLUMNS`).

```bash
./timing.py test/data/runs/Basalt/TR5/timing.csv --plot
```

### Features

See the number of features each pose was computed with.
It supports comparing multiple files as well.

```bash
./features.py --plot test/data/runs/Basalt/EMH02/features.csv
```

### Completion

See what percentage of the dataset the run was able to complete without crashing.

```bash
./completion.py test/data/runs/Kimera/TR5/tracking.csv test/data/targets/TR5/cam0.csv
```

### Tracking error

See ATE, RTE, or SDM stats for a particular run.

```bash
./tracking.py ate test/data/targets/EV202/gt.csv test/data/runs/Basalt/EV202/tracking.csv --plot --plot_mode xyz

# Or if you want to compare multiple trajectories
./tracking.py ate test/data/targets/EMH02/gt.csv test/data/runs/Basalt/EMH02/tracking.csv test/data/runs/ORB-SLAM3/EMH02/tracking.csv --plot --plot_mode xy
```

### Batch comparison

Generate tables comparing averages of multiple runs on multiple datasets. Take
notice of the `runs` and `targets` directory structures and of the fact that you
need to specify the start/end timing column for each run name. The latter
will be fixed once standard start/end column names are in place.

```bash
./batch.py test/data/runs/ test/data/targets/
```

`batch.py` expects the `targets` directory to have camera timestamps `cam0.csv`
and optionally groundtruth `gt.csv` files that you can get from the datasets
themselves. To ease things a bit, you can uncompress
`tar -xvf test/data/targets.tar.xz -C test/data/` to get those files for all EuRoC,
TUM-VI room, and [our custom (without
groundtruth)](https://bit.ly/monado-datasets) inside `test/data/targets`.

### EuRoC tools

The script `euroc/euroc_ops.py` contains multiple commands for operating on EuRoC datasets:

- `imu2cam_ts`: Applies a time offset to all IMU samples. Read code.
- `get_duration`: Print duration of the dataset.
- `verify`: Perform a number of checks on the dataset to verify it's a valid one.
- `get_max_sensor_dt`: Get the maximum deltatime between samples of one of the dataset's sensors.
- `cam_offset_ts`: Create a new camera csv file with its timestamps offseted by some delta nanoseconds from input.
- `trim`: Produce a trimmed-down copy of the dataset between two timestamps (or rather, _seconds_ into the dataset) specified as input.
- `apply_imu_calib`: Given a basalt calibration file, it precalibrates all IMU samples; i.e., it applies the IMU [mixing matrix and biases](https://vladyslavusenko.gitlab.io/basalt-headers/classbasalt_1_1CalibGyroBias.html#details) to all samples.
- `preview_video`: It generates a preview video from a given dataset using `ffmpeg`.
