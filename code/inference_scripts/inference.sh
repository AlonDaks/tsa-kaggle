#!/bin/bash

# ./create_png.sh
# ./keypoint_inference.sh

echo 'Id,Probability' > kaggle_submission_stage_2.csv 
python3 ../tsa/modeling/models/inference.py --data_set stage2 --region arm   >> kaggle_submission_stage_2.csv 2>/dev/null
python3 ../tsa/modeling/models/inference.py --data_set stage2 --region torso >> kaggle_submission_stage_2.csv 2>/dev/null
python3 ../tsa/modeling/models/inference.py --data_set stage2 --region thigh >> kaggle_submission_stage_2.csv 2>/dev/null
python3 ../tsa/modeling/models/inference.py --data_set stage2 --region calf  >> kaggle_submission_stage_2.csv 2>/dev/null