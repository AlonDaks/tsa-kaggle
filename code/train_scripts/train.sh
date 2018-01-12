#!/bin/bash

./create_png.sh
./create_keypoint_tf_records.sh
./train_keypoints.sh
./keypoint_inference.sh
./create_tf_records.sh
./train_arm.sh
./train_calf.sh
./train_thigh.sh
./train_torso.sh
./train_localization_nets.sh
./curate_arc_calibration.sh