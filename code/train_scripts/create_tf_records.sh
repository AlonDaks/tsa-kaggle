#!/bin/bash

## Create Curriculum
python3 ../tsa/modeling/models/create_tf_record.py --data_set train_0 --file_format aps --region calf --random_seed 42 --mode pure &
python3 ../tsa/modeling/models/create_tf_record.py --data_set train_0 --file_format aps --region calf --random_seed 43 --mode mask &
python3 ../tsa/modeling/models/create_tf_record.py --data_set validate_0 --file_format aps --region calf --random_seed 52 --mode pure &
python3 ../tsa/modeling/models/create_tf_record.py --data_set validate_0 --file_format aps --region calf --random_seed 53 --mode mask &

python3 ../tsa/modeling/models/create_tf_record.py --data_set train_0 --file_format aps --region thigh --random_seed 44 --mode pure &
python3 ../tsa/modeling/models/create_tf_record.py --data_set train_0 --file_format aps --region thigh --random_seed 45 --mode mask &
python3 ../tsa/modeling/models/create_tf_record.py --data_set validate_0 --file_format aps --region thigh --random_seed 54 --mode pure &
python3 ../tsa/modeling/models/create_tf_record.py --data_set validate_0 --file_format aps --region thigh --random_seed 55 --mode mask &

python3 ../tsa/modeling/models/create_tf_record.py --data_set train_0 --file_format aps --region arm --random_seed 46 --mode pure &
python3 ../tsa/modeling/models/create_tf_record.py --data_set train_0 --file_format aps --region arm --random_seed 47 --mode mask &
python3 ../tsa/modeling/models/create_tf_record.py --data_set validate_0 --file_format aps --region arm --random_seed 56 --mode pure &
python3 ../tsa/modeling/models/create_tf_record.py --data_set validate_0 --file_format aps --region arm --random_seed 57 --mode mask &

python3 ../tsa/modeling/models/create_tf_record.py --data_set train_0 --file_format aps --region torso --random_seed 48 --mode pure &
python3 ../tsa/modeling/models/create_tf_record.py --data_set train_0 --file_format aps --region torso --random_seed 49 --mode mask &
python3 ../tsa/modeling/models/create_tf_record.py --data_set validate_0 --file_format aps --region torso --random_seed 58 --mode pure &
python3 ../tsa/modeling/models/create_tf_record.py --data_set validate_0 --file_format aps --region torso --random_seed 59 --mode mask &

python3 ../tsa/modeling/models/create_tf_record.py --data_set train_1 --file_format aps --region thigh --random_seed 54 --mode pure &
python3 ../tsa/modeling/models/create_tf_record.py --data_set train_1 --file_format aps --region thigh --random_seed 55 --mode mask &
python3 ../tsa/modeling/models/create_tf_record.py --data_set validate_1 --file_format aps --region thigh --random_seed 54 --mode pure &
python3 ../tsa/modeling/models/create_tf_record.py --data_set validate_1 --file_format aps --region thigh --random_seed 55 --mode mask &

# Create localization
python3 ../tsa/modeling/models/create_tf_record.py --data_set train_0 --region arm --random_seed 29 --mode localization &
python3 ../tsa/modeling/models/create_tf_record.py --data_set validate_0 --region arm --random_seed 29 --mode localization &
python3 ../tsa/modeling/models/create_tf_record.py --data_set train_0 --region thigh --random_seed 29 --mode localization &
python3 ../tsa/modeling/models/create_tf_record.py --data_set validate_0 --region thigh --random_seed 29 --mode localization &