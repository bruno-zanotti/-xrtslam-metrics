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

### Completion

See what percentage of the dataset the run was able to complete without crashing.

```bash
./completion.py test/data/runs/Kimera/TR5/tracking.csv test/data/targets/TR5/cam0.csv
```

### Tracking error

See ATE or RTE stats for a particular run.

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
