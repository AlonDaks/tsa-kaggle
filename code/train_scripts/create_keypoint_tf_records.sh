#!/bin/bash

python3 ../tsa/modeling/models/create_tf_record.py --data_set train_0 --region face --random_seed 29 &
python3 ../tsa/modeling/models/create_tf_record.py --data_set validate_0 --region face --random_seed 29 &

python3 ../tsa/modeling/models/create_tf_record.py --data_set train_0 --region butt --random_seed 29 &
python3 ../tsa/modeling/models/create_tf_record.py --data_set validate_0 --region butt --random_seed 29 &