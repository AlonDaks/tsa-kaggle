#!/bin/bash

python3 ../tsa/modeling/models/threat_detect_net.py --train_dir calf_pure --mode pure --region calf --fold 0 --mean_subtract 8.5 --file_format aps --max_steps 35000 --random_seed 90
python3 ../tsa/modeling/models/threat_detect_net.py --train_dir calf_mask --mode mask --region calf --fold 0 --mean_subtract 8.5 --file_format aps --max_steps 30000 --random_seed 91 --checkpoint_dir calf_pure
python3 ../tsa/modeling/models/threat_detect_net.py --train_dir calf_full --mode full --region calf --fold 1 --mean_subtract 8.5 --file_format aps --max_steps 11000 --random_seed 92 --checkpoint_dir calf_mask
