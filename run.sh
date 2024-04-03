#!/bin/bash
module load use.own
module load conda-env/mypackages-py3.8.5

epochs=300
ds="/scratch/gilbreth/wweligam/FaceData/train_faces"
args="--label_noise 0.0 --validate_every 10 --refresh_dataset_every 25 --val_ratio 0.2 --test_ratio 0.002  --loss_image 1*L1 --loss_label 1*CrossEntropyLoss"
python main_train.py --model inc2 --dataset_path $ds --epochs $epochs $args --batch_size 64 &
sleep 2
python main_train.py --model inc --dataset_path $ds --epochs $epochs $args --batch_size 64 &
sleep 2
python main_train.py --model model1 --dataset_path $ds --epochs $epochs $args --batch_size 64 &
sleep 2
python main_train.py --model model2 --dataset_path $ds --epochs $epochs $args --batch_size 64 &
sleep 2

#python main_train.py --model model1_small --dataset_path $ds --epochs $epochs $args --batch_size 64 &
#sleep 2
#python main_train.py --model model2_small --dataset_path $ds --epochs $epochs $args --batch_size 64 &
#sleep 2
#python main_train.py --model simple --dataset_path $ds --epochs $epochs $args --batch_size 64 &
#sleep 2
#python main_train.py --model simple_small --dataset_path $ds --epochs $epochs $args --batch_size 64 &
#sleep 2
#python main_train.py --model simple_medium --dataset_path $ds --epochs $epochs $args --batch_size 64 &
#sleep 2
#python main_train.py --model simple_medium2 --dataset_path $ds --epochs $epochs $args --batch_size 64 &
#sleep 2