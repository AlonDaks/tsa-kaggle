#!/bin/bash

python3 ../tsa/utils/create_png.py --data_set train_0 --file_format aps
python3 ../tsa/utils/create_png.py --data_set validate_0 --file_format aps

python3 ../tsa/utils/create_png.py --data_set train_0 --file_format a3daps
python3 ../tsa/utils/create_png.py --data_set validate_0 --file_format a3daps