#!/bin/bash
module load use.own
module load conda-env/mypackages-py3.8.5

rm /scratch/gilbreth/wweligam/FaceData/test_faces/*
python preprocess_data.py --batch_size 64  --dataset_path "/scratch/gilbreth/wweligam/FaceData/test"

rm /scratch/gilbreth/wweligam/FaceData/train_small_faces/*
python preprocess_data.py --batch_size 64  --dataset_path "/scratch/gilbreth/wweligam/FaceData/train_small"

rm /scratch/gilbreth/wweligam/FaceData/train_faces/*
python preprocess_data.py --batch_size 64  --dataset_path "/scratch/gilbreth/wweligam/FaceData/train"