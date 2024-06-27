#!/usr/bin/env bash

format_trajectory_file() {
    local trajectory_file=$1

    # Rename file
    new_file="trajectory.csv"

    # Add headers
    headers="#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []"

    # Replace spaces by commas and remove decimals from the first column
    {
        echo "$headers"
        sed -e 's/ /,/g' -e 's/\([0-9]*\)\.[0-9]*,/\1,/' "$trajectory_file"
    } > temp && mv temp "$new_file"

}

run_orb_slam3() {
    local dataset=$1
    local dataset_path=$2
    local config_path=$3
    local timestamps_path=$4
    local show_gui=$5
    local timestamp_file="$timestamps_path/${dataset}.txt"
    local trajectory_file="trajectory"

    mkdir -p "$ROOT/$RUN/$dataset" && cd "$ROOT/$RUN/$dataset"
    printf "%s " $dataset

    cmd="$ORBSLAM3_DIR/Examples/Stereo-Inertial/stereo_inertial_euroc $ORBSLAM3_DIR/Vocabulary/ORBvoc.txt $config_path $dataset_path $timestamp_file $trajectory_file"
    start_time=$(date +%s.%N)
    eval "$cmd" &> "$ROOT/$RUN/$dataset/output.log"
    if [ $? -ne 0 ]; then
        echo "Segmentation fault in dataset $dataset. Check output.log for details."
    fi
    end_time=$(date +%s.%N)
    diff=$(echo $end_time - $start_time | bc -l)
    awk -v start=$start_time -v end=$end_time 'BEGIN { diff = end - start; printf "%.2f seconds\n", diff }'

    # Format trajectory file
    format_trajectory_file "f_${trajectory_file}.txt"

}

SCRIPT_DIR=$(dirname "$(realpath "$0")")
ROOT="$SCRIPT_DIR/.."
RUN=$(basename "$SCRIPT_DIR")
source $ROOT/settings.cfg

# ORB-SLAM3
ORBSLAM3_DIR=~/tesina/orb_slam3

# DATASETS
DATASETS_DIR=~/tesina/datasets
EUROC_DIR=$DATASETS_DIR/euroc
TUM_DIR=$DATASETS_DIR/tum
MSDMI_DIR=$DATASETS_DIR/msdmi
MSDMG_DIR=$DATASETS_DIR/msdmg
MSDMO_DIR=$DATASETS_DIR/msdmo

# CALIB
# CALIB_DIR=$ROOT/_calibs
# euroc_calib=$CALIB_DIR/euroc.json
# msdmi_calib=$CALIB_DIR/msdmi.json
# msdmg_calib=$CALIB_DIR/msdmg.json

# CONFIG
CONFIG_DIR=$ROOT/$RUN/_configs
euroc_config=$CONFIG_DIR/euroc.yaml
tumvi_config=$CONFIG_DIR/tumvi.yaml
# msdmi_config=$CONFIG_DIR/msdmi.json
# msdmg_config=$CONFIG_DIR/msdmg.json
msdmo_config=$CONFIG_DIR/msdmo.yaml

# TIMESTAMPS
msdmo_timestamps="$ORBSLAM3_DIR/Examples/Stereo-Inertial/msdmo_timestamps"

rm -f faillist
rm -f startfinish

date +%s >> startfinish

# Run ORB-SLAM3 commands
echo Running $RUN:

# EUROC
for dataset in "${euroc_datasets[@]}"; do
    dataset_path=$EUROC_DIR/$dataset
    run_orb_slam3 $dataset $dataset_path $euroc_config $show_gui
done

# MSDMI
for dataset in "${msdmi_datasets[@]}"; do
    dataset_path=$MSDMI_DIR/$dataset
    run_orb_slam3 $dataset $dataset_path $msdmi_config $show_gui
done

# MSDMG
for dataset in "${msdmg_datasets[@]}"; do
    dataset_path=$MSDMG_DIR/$dataset
    run_orb_slam3 $dataset $dataset_path $msdmg_config $show_gui
done

# MSDMO
for dataset in "${msdmo_datasets[@]}"; do
    dataset_path=$MSDMO_DIR/$dataset
    run_orb_slam3 $dataset $dataset_path $msdmo_config $msdmo_timestamps $show_gui
done

date +%s >> startfinish
paplay /usr/share/sounds/freedesktop/stereo/complete.oga &
