#!/bin/bash
module load use.own
module load conda-env/mypackages-py3.8.5

rm ./test_faces/*
python preprocess_data.py --batch_size 64  --dataset_path "/scratch/gilbreth/wweligam/FaceData/test"

rm ./train_small_faces/*
python preprocess_data.py --batch_size 64  --dataset_path "/scratch/gilbreth/wweligam/FaceData/train_small"

rm ./train_faces/*
python preprocess_data.py --batch_size 64  --dataset_path "/scratch/gilbreth/wweligam/FaceData/train"