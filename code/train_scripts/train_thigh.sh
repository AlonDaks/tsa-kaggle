#!/bin/bash

python3 ../tsa/modeling/models/threat_detect_net.py --train_dir thigh_pure --mode pure --region thigh --fold 0 --mean_subtract 35 --file_format aps --max_steps 35000 --random_seed 80
python3 ../tsa/modeling/models/threat_detect_net.py --train_dir thigh_mask --mode mask --region thigh --fold 0 --mean_subtract 35 --file_format aps --max_steps 28000 --random_seed 81 --checkpoint_dir thigh_pure
python3 ../tsa/modeling/models/threat_detect_net.py --train_dir thigh_full --mode full --region thigh --fold 1 --mean_subtract 35 --file_format aps --max_steps 10000 --random_seed 82 --checkpoint_dir thigh_mask
