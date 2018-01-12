#!/bin/bash

python3 ../tsa/modeling/models/curate_arc_calibration.py --data_set train_0 --region arm
python3 ../tsa/modeling/models/curate_arc_calibration.py --data_set validate_0 --region arm

python3 ../tsa/modeling/models/curate_arc_calibration.py --data_set train_0 --region torso
python3 ../tsa/modeling/models/curate_arc_calibration.py --data_set validate_0 --region torso

python3 ../tsa/modeling/models/curate_arc_calibration.py --data_set train_0 --region thigh
python3 ../tsa/modeling/models/curate_arc_calibration.py --data_set validate_0 --region thigh

python3 ../tsa/modeling/models/curate_arc_calibration.py --data_set train_0 --region calf
python3 ../tsa/modeling/models/curate_arc_calibration.py --data_set validate_0 --region calf