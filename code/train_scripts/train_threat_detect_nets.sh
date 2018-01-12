# Curriculum Training
python3 ../tsa/modeling/models/threat_detect_net.py --train_dir torso_pure --mode pure --region torso --fold 0 --mean_subtract 23.4 --file_format aps --max_steps 35000 --random_seed 60
python3 ../tsa/modeling/models/threat_detect_net.py --train_dir torso_mask --mode mask --region torso --fold 0 --mean_subtract 23.4 --file_format aps --max_steps 45000 --random_seed 61 --checkpoint_dir torso_pure
python3 ../tsa/modeling/models/threat_detect_net.py --train_dir torso_full --mode full --region torso --fold 1 --mean_subtract 23.4 --file_format aps --max_steps 20000 --random_seed 62 --checkpoint_dir torso_mask

python3 ../tsa/modeling/models/threat_detect_net.py --train_dir arm_pure --mode pure --region arm --fold 0 --mean_subtract 5.5 --file_format aps --max_steps 35000 --random_seed 70
python3 ../tsa/modeling/models/threat_detect_net.py --train_dir arm_mask --mode mask --region arm --fold 0 --mean_subtract 5.5 --file_format aps --max_steps 45000 --random_seed 71 --checkpoint_dir arm_pure
python3 ../tsa/modeling/models/threat_detect_net.py --train_dir arm_full --mode full --region arm --fold 1 --mean_subtract 5.5 --file_format aps --max_steps 20000 --random_seed 72 --checkpoint_dir arm_mask

python3 ../tsa/modeling/models/threat_detect_net.py --train_dir thigh_pure --mode pure --region thigh --fold 0 --mean_subtract 35 --file_format aps --max_steps 35000 --random_seed 80
python3 ../tsa/modeling/models/threat_detect_net.py --train_dir thigh_mask --mode mask --region thigh --fold 0 --mean_subtract 35 --file_format aps --max_steps 28000 --random_seed 81 --checkpoint_dir thigh_pure
python3 ../tsa/modeling/models/threat_detect_net.py --train_dir thigh_full --mode full --region thigh --fold 1 --mean_subtract 35 --file_format aps --max_steps 10000 --random_seed 82 --checkpoint_dir thigh_mask

python3 ../tsa/modeling/models/threat_detect_net.py --train_dir calf_pure --mode pure --region calf --fold 0 --mean_subtract 8.5 --file_format aps --max_steps 35000 --random_seed 90
python3 ../tsa/modeling/models/threat_detect_net.py --train_dir calf_mask --mode mask --region calf --fold 0 --mean_subtract 8.5 --file_format aps --max_steps 30000 --random_seed 91 --checkpoint_dir calf_pure
python3 ../tsa/modeling/models/threat_detect_net.py --train_dir calf_full --mode full --region calf --fold 1 --mean_subtract 8.5 --file_format aps --max_steps 11000 --random_seed 92 --checkpoint_dir calf_mask
