#!/bin/bash

python3 ../tsa/modeling/models/localization_net.py --train_dir arm_localization --region arm --mean_subtract 5.5  --max_steps 12000 --random_seed 29
python3 ../tsa/modeling/models/localization_net.py --train_dir thigh_localization --region thigh  --mean_subtract 35  --max_steps 12000 --random_seed 29
