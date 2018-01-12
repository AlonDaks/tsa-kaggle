#!/bin/bash

python3 ../tsa/modeling/models/threat_detect_net.py --train_dir torso_pure --mode pure --region torso --fold 0 --mean_subtract 23.4 --file_format aps --max_steps 35000 --random_seed 60
python3 ../tsa/modeling/models/threat_detect_net.py --train_dir torso_mask --mode mask --region torso --fold 0 --mean_subtract 23.4 --file_format aps --max_steps 45000 --random_seed 61 --checkpoint_dir torso_pure
python3 ../tsa/modeling/models/threat_detect_net.py --train_dir torso_full --mode full --region torso --fold 1 --mean_subtract 23.4 --file_format aps --max_steps 10000 --random_seed 62 --checkpoint_dir torso_mask
