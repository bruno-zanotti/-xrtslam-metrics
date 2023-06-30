#!/usr/bin/env bash

export EUROC_MAX_SPEED=off
export SHOW_GUI=1
export INSTALL_DIR=/usr/local/etc/basalt

export euroc_config=$INSTALL_DIR/euroc.toml
export tumvi_config=$INSTALL_DIR/tumvi.toml
export d640_config=$INSTALL_DIR/d455-640x480.toml
export d848_config=$INSTALL_DIR/d455-848x480.toml
export ody_config=$INSTALL_DIR/odysseyplus_rt8.toml
# export ody_config=$INSTALL_DIR/odysseyplus_kb4.toml

if [ -w $INSTALL_DIR ]; then
  sed -i "s/show-gui=./show-gui=$SHOW_GUI/" $INSTALL_DIR/*.toml
else
  sudo sed -i "s/show-gui=./show-gui=$SHOW_GUI/" $INSTALL_DIR/*.toml
fi

date +%s >> startfinish
monado-cli slambatch $euroc/MH_01_easy/           $euroc_config EMH01 || echo "EMH01"  >> faillist
monado-cli slambatch $euroc/MH_02_easy/           $euroc_config EMH02 || echo "EMH02"  >> faillist
monado-cli slambatch $euroc/MH_03_medium/         $euroc_config EMH03 || echo "EMH03"  >> faillist
monado-cli slambatch $euroc/MH_04_difficult/      $euroc_config EMH04 || echo "EMH04"  >> faillist
monado-cli slambatch $euroc/MH_05_difficult/      $euroc_config EMH05 || echo "EMH05"  >> faillist
monado-cli slambatch $euroc/V1_01_easy/           $euroc_config EV101 || echo "EV101"  >> faillist
monado-cli slambatch $euroc/V1_02_medium/         $euroc_config EV102 || echo "EV102"  >> faillist
monado-cli slambatch $euroc/V1_03_difficult/      $euroc_config EV103 || echo "EV103"  >> faillist
monado-cli slambatch $euroc/V2_01_easy/           $euroc_config EV201 || echo "EV201"  >> faillist
monado-cli slambatch $euroc/V2_02_medium/         $euroc_config EV202 || echo "EV202"  >> faillist
monado-cli slambatch $tumvi/dataset-room1_512_16/ $tumvi_config TR1   || echo "TR1"    >> faillist
monado-cli slambatch $tumvi/dataset-room2_512_16/ $tumvi_config TR2   || echo "TR2"    >> faillist
monado-cli slambatch $tumvi/dataset-room3_512_16/ $tumvi_config TR3   || echo "TR3"    >> faillist
monado-cli slambatch $tumvi/dataset-room4_512_16/ $tumvi_config TR4   || echo "TR4"    >> faillist
monado-cli slambatch $tumvi/dataset-room5_512_16/ $tumvi_config TR5   || echo "TR5"    >> faillist
monado-cli slambatch $tumvi/dataset-room6_512_16/ $tumvi_config TR6   || echo "TR6"    >> faillist
monado-cli slambatch $custom/d455-640-easy/       $d640_config C6EASY || echo "C6EASY" >> faillist
monado-cli slambatch $custom/d455-640-hard/       $d640_config C6HARD || echo "C6HARD" >> faillist
monado-cli slambatch $custom/d455-848-easy/       $d848_config C8EASY || echo "C8EASY" >> faillist
monado-cli slambatch $custom/d455-848-hard/       $d848_config C8HARD || echo "C8HARD" >> faillist
monado-cli slambatch $custom/ody-easy/            $ody_config  COEASY || echo "COEASY" >> faillist
monado-cli slambatch $custom/ody-hard/            $ody_config  COHARD || echo "COHARD" >> faillist
date +%s >> startfinish
paplay /usr/share/sounds/freedesktop/stereo/complete.oga &
