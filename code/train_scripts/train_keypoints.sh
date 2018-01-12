#!/bin/bash

python3 ../../code/tsa/modeling/models/keypoint_inference.py --train_dir face_keypoint --keypoint face --max_steps 12000 --random_seed 29
python3 ../../code/tsa/modeling/models/keypoint_inference.py --train_dir butt_keypoint --keypoint butt --max_steps 12000 --random_seed 29