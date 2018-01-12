# tsa-kaggle

This repository contains code for solving [Kaggle's Passanger Screening Algorithm Challenge](kaggle.com/c/passenger-screening-algorithm-challenge) sponsored by the Department of Homeland Security. The project ranked 33rd / 518 and can be found on the [leaderboard](https://www.kaggle.com/c/passenger-screening-algorithm-challenge/leaderboard) under team Alon Daks.

## Contact
Please direct any questions or clarifications to the author: Alon Daks, hello@alondaks.com.

## Note on Data
The rules of the Kaggle competition prohibit public sharing of the body image/scan dataset. This repository, therefore, does not contain any source data provided by DHS.

## Hardware Dependencies
This package requires an nvidia GPU with ~12 GB of memory in order to efficiently run training and inference code. It was developed on a machine equipt with a current generation 1080 ti. It also requires about 1TB of available storage.

The code depends on tensorflow-gpu 1.4.0, which requires Cuda Toolkit 8.0 and cudNN 6.0. See https://www.tensorflow.org/install/install_linux for more details.

## Installation

* python 3.5.2 (virtual env recommended)

* python packages (and versions) listed in `requirements.txt`
  * install with `pip install -r requirements.txt`
* `tsa` package contained in this repository
  * install by adding the `code/` directory to your PYTHONPATH
  * e.g. `export PYTHONPATH='<path to project repo>/tsa-kaggle-2/code'`
* designating a directory on disk for storing large data
  * since raw data and tf_records are too large to contain in this repository, the code requires an external directory for storing these large files. The code accesses this directory via the `TSA_KAGGLE_DATA_DIR` environment variable. Ensure `TSA_KAGGLE_DATA_DIR` is set to a directory that can hold about 1TB of data
  * e.g. `export TSA_KAGGLE_DATA_DIR/home/alon/sdb1/tsa-kaggle`
  * The code assumes a structured directory tree sitting inside `TSA_KAGGLE_DATA_DIR`. Execute the follow command to build this tree: `mkdir -p $TSA_KAGGLE_DATA_DIR/{data/{raw/{aps,a3daps,aps_png/{0..15},a3daps_png/{0}},tf_records/},train_dir/}`
  
## Data Placement for Stage 2
* place aps and a3daps raw data files inside their respective directories in `$TSA_KAGGLE_DATA_DIR/data/raw`. This must be done prior to running stage-2 inference
* create a file called `stage2_ids.csv` inside `tsa-kaggle-2/data/` containing the stage2 ids. This file should match the structure of `tsa-kaggle-2/data/train_0_ids.csv` (e.g. same header: `Id`)

## Running Training
* to run training, run `./train.sh` from within the `tsa-kaggle-2/code/train_scripts`. This will probably take around 3 days to compute on the hardware described above
* Note that since tensorflow has non-deterministic GPU implementations for computing gradients, it might not be possible to reproduce the exact same weights I computed, but these should be close.
* I am also including the weights themselves in the upload

## Running stage2 inference
* run `./inference.sh` from within the `tsa-kaggle-2/inference_scripts` directory
* this script assumes all the checkpoints / saved training files already exist

## Model Weight Checksums
Since the model weights are too large to upload, I am including each of the checksums here for those weights used during inference. These checksums are computed using sha256. 

To reproduce my exact inference predictions, the files corresponding to these checksums are needed. I will be happy to share them if necessary.

- `$TSA_KAGGLE_DATA_DIR/train_dir/arm_full.zip`: `e6e38f4a2a57608817fcc947cd24a9d40531ef56cda4d2b5008c63081257fe7b`
- `$TSA_KAGGLE_DATA_DIR/train_dir/torso_full.zip`: `f1189a48efc8b5722ccd6bcfc613fdb55315b37cae1f8a6e299585d4b2a027f2`
- `$TSA_KAGGLE_DATA_DIR/train_dir/thigh_full.zip`: `eb01e0ebded206425c9101a2f721a8fdd9b419c668a8864f86cb28a9e9a4630e`
- `$TSA_KAGGLE_DATA_DIR/train_dir/calf_full.zip`: `06498eb8d795ebeb3728e4f86ad03c55be1f73dcddff62b37e4708adb084411c`

- `$TSA_KAGGLE_DATA_DIR/train_dir/face_keypoint.zip`: `e95064698c4fcd0e398f59e643f206f009ed3b1ff2b4432eb62c838fb578dc32`
- `$TSA_KAGGLE_DATA_DIR/train_dir/butt_keypoint.zip`: `ac2cb544fd757202b6d38703e09b04410c7aac57fa79e39bce959531bbf9117b`

- `$TSA_KAGGLE_DATA_DIR/train_dir/arm_localization.zip`: `83c950a6edd415a5e4d504f13aa9ec83ae7781c9e25f5520f0fc9d2f3f72cdc5`
- `$TSA_KAGGLE_DATA_DIR/train_dir/thigh_localization.zip`: `39a1f8fa6d000338733dca66b380408105f8de20453b678ca0bd05aab000fedb`

