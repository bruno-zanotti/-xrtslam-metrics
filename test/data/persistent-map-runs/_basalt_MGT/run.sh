#!/usr/bin/env bash

# TODO: Add --use-imu 0
run_basalt_vio() {
    local dataset=$1
    local dataset_path=$2
    local cam_calib=$3
    local config_path=$4
    local show_gui=$5

    mkdir -p "$ROOT/$RUN/$dataset" && cd "$ROOT/$RUN/$dataset"
    printf "%s " $dataset
    start_time=$(date +%s.%N)
    if basalt_vio --dataset-path $dataset_path \
                  --cam-calib $cam_calib \
                  --dataset-type euroc \
                  --config-path $config_path \
                  --show-gui $show_gui \
                  --save-trajectory euroc \
                  --save-features 1 \
       > /dev/null; then
        end_time=$(date +%s.%N)
        diff=$(echo $end_time - $start_time | bc -l)
        awk -v start=$start_time -v end=$end_time 'BEGIN { diff = end - start; printf "%.2f seconds\n", diff }'
    else
        echo $dataset >> $ROOT/$RUN/faillist
    fi
}

SCRIPT_DIR=$(dirname "$(realpath "$0")")
ROOT="$SCRIPT_DIR/.."
RUN=$(basename "$SCRIPT_DIR")
source $ROOT/settings.cfg

# DATASETS
DATASETS_DIR=~/tesina/datasets
EUROC_DIR=$DATASETS_DIR/euroc
TUM_DIR=$DATASETS_DIR/tum
MSDMI_DIR=$DATASETS_DIR/msdmi

# CALIB
CALIB_DIR=$ROOT/_calibs
euroc_calib=$CALIB_DIR/euroc.json
msdmi_calib=$CALIB_DIR/msdmi.json

# CONFIG
CONFIG_DIR=$ROOT/$RUN/_configs
euroc_config=$CONFIG_DIR/euroc.json
msdmi_config=$CONFIG_DIR/msdmi.json

rm -f faillist
rm -f startfinish

date +%s >> startfinish

# Run Basalt commands
echo Running $RUN:

# EUROC
for dataset in "${euroc_datasets[@]}"; do
    dataset_path=$EUROC_DIR/$dataset
    run_basalt_vio $dataset $dataset_path $euroc_calib $euroc_config $show_gui
done

# MSDMI
for dataset in "${msdmi_datasets[@]}"; do
    dataset_path=$MSDMI_DIR/$dataset
    run_basalt_vio $dataset $dataset_path $msdmi_calib $msdmi_config $show_gui
done

date +%s >> startfinish
paplay /usr/share/sounds/freedesktop/stereo/complete.oga &
