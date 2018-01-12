#!/bin/bash

python3 ../tsa/modeling/models/keypoint_inference.py --data_set train_0 --keypoint face
python3 ../tsa/modeling/models/keypoint_inference.py --data_set validate_0 --keypoint face

python3 ../tsa/modeling/models/keypoint_inference.py --data_set train_0 --keypoint butt
python3 ../tsa/modeling/models/keypoint_inference.py --data_set validate_0 --keypoint butt
