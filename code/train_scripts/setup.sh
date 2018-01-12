#!/bin/bash

# sudo mkfs.ext4 /dev/xvdb
# mkdir /home/ubuntu/storage_volume
# sudo mount /dev/xvdb /home/ubuntu/storage_volume
# sudo chown ubuntu /home/ubuntu/storage_volume

sudo apt-get update

sudo apt install python3
echo "alias python='/usr/bin/python3'" >> ~/.bashrc
alias python='/usr/bin/python3'
sudo apt-get install python3-pip

# Install cuda
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
mv cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda

wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/patches/2/cuda-repo-ubuntu1604-8-0-local-cublas-performance-update_8.0.61-1_amd64-deb
mv cuda-repo-ubuntu1604-8-0-local-cublas-performance-update_8.0.61-1_amd64-deb cuda-repo-ubuntu1604-8-0-local-cublas-performance-update_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-cublas-performance-update_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda

echo 'export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc

source ~/.bashrc

rm cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
rm cuda-repo-ubuntu1604-8-0-local-cublas-performance-update_8.0.61-1_amd64.deb


# Install cuDNN
tar -xvf cudnn-8.0-linux-x64-v6.0.tgz -C ~
pushd ~
sudo mkdir -p /usr/local/cuda-8.0/{include,lib64}
sudo cp cuda/include/cudnn.h /usr/local/cuda-8.0/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64
sudo chmod a+r /usr/local/cuda-8.0/include/cudnn.h /usr/local/cuda-8.0/lib64/libcudnn*
popd

sudo pip3 install -r ../../requirements.txt

echo "export PYTHONPATH='/home/ubuntu/tsa-kaggle-2/code'" >> ~/.bashrc
echo 'export TSA_KAGGLE_DATA_DIR=/home/ubuntu/storage_volume' >> ~/.bashrc

source ~/.bashrc

mkdir -p $TSA_KAGGLE_DATA_DIR/{data/{raw/stage1/{aps,a3daps,aps_png/{0..15},a3daps_png/{0}},tf_records/},train_dir/}


# git clone https://github.com/AlonDaks/tsa-kaggle-2.git
#scp -i tsa-kaggle.pem cudnn-8.0-linux-x64-v6.0.tgz ubuntu@ec2-54-202-23-49.us-west-2.compute.amazonaws.com:~/tsa-kaggle-2/code/scripts
# scp -i ~/Desktop/tsa-kaggle.pem stage1_aps.tar.gz ubuntu@ec2-54-202-161-235.us-west-2.compute.amazonaws.com:/home/ubuntu/storage_volume/data/raw/stage1
# scp -i ~/Desktop/tsa-kaggle.pem vgg16.npz ubuntu@ec2-54-202-161-235.us-west-2.compute.amazonaws.com:/home/ubuntu/tsa-kaggle-2/data/pretrained_weights/vgg16.npz