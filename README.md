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

See a plot of the times each pose from the dataset took to compute. Needs the
`timing.csv` generated from the run, and the start/end column names you want to
compare.

```bash
./timing.py test/data/runs/Basalt/TR5/timing.csv tracker_pushed vio_produced --plot
```

### Completion

See what percentage of the dataset the run was able to complete without crashing.

```bash
./completion.py test/data/runs/Kimera/TR5/tracking.csv test/data/targets/TR5/cam0.csv
```

### Tracking error

See ATE or RTE stats for a particular run.

```bash
./tracking.py ate test/data/runs/Basalt/EV202/tracking.csv test/data/targets/EV202/gt.csv --plot
```

### Batch comparison

Generate tables comparing averages of multiple runs on multiple datasets. Take
notice of the `runs` and `targets` directory structures and of the fact that you
need to specify the start/end timing column for each run name. The latter
will be fixed once standard start/end column names are in place.

```bash
./batch.py test/data/runs/ test/data/targets/ \
  --timing Basalt opticalflow_received vio_produced \
  --timing Kimera tracker_pushed processed \
  --timing ORB-SLAM3 about_to_process processed
```

`batch.py` expects the `targets` directory to have camera timestamps `cam0.csv`
and optionally groundtruth `gt.csv` files that you can get from the datasets
themselves. To ease things a bit, you can uncompress `tar -xvf
test/data/targets-full.tar.xz -C test/data/` to get those files for all EuRoC,
TUM-VI room, and [our custom (without
groundtruth)](https://bit.ly/monado-datasets) inside `test/data/targets`.
