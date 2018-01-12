#!/bin/bash

python3 ../tsa/modeling/models/threat_detect_net.py --train_dir arm_pure --mode pure --region arm --fold 0 --mean_subtract 5.5 --file_format aps --max_steps 20000 --random_seed 70
python3 ../tsa/modeling/models/threat_detect_net.py --train_dir arm_mask --mode mask --region arm --fold 0 --mean_subtract 5.5 --file_format aps --max_steps 30000 --random_seed 71 --checkpoint_dir arm_pure
python3 ../tsa/modeling/models/threat_detect_net.py --train_dir arm_full --mode full --region arm --fold 1 --mean_subtract 5.5 --file_format aps --max_steps 10000 --random_seed 72 --checkpoint_dir arm_mask
