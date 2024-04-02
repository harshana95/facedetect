import argparse
import glob
import os

import cv2
import einops
import numpy as np
import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from dataset.BasicDataset import ImageDataset
from facenet_pytorch import MTCNN, InceptionResnetV1
from utils.dataset_utils import resize, fix_image_shape, padding
from utils.default_args import add_default_args, parse_args
from utils.utils import generate_folder


def extract_faces(image, m):
    # Get cropped and prewhitened image tensor
    img_cropped, prob = m(image, return_prob=True)
    return img_cropped, prob


def try_bad_image(image, mtcnn):
    crop, probs = extract_faces(image, mtcnn)
    h, w = image.shape[:2]
    _image = image.detach().clone()
    if crop is None:
        _image[:, :w // 2] = image[:, w // 2:]
        _image[:, w // 2:] = image[:, :w // 2]
        crop, probs = extract_faces(_image, mtcnn)
    if crop is None:
        pad = padding(h * 2, w * 2, [])
        _image = pad(_image)
        _image = torch.tensor(cv2.resize(_image.numpy(), (w, h)))
        crop, probs = extract_faces(_image, mtcnn)
    return crop, probs


def save_face(new_labels, image, label, fname, j, save_loc):
    new_filename = f"{fname.split('.')[0]}_{j:03d}.{fname.split('.')[1]}"
    new_labels.loc[len(new_labels)] = [new_filename, label]
    plt.imsave(os.path.join(save_loc, new_filename), image)
    return new_labels


if __name__ == "__main__":
    datasetname = 'train_small'
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser, 'face_detect', 64, datasetname=datasetname)
    args = parse_args(parser)

    skip_multiple_faces = False
    save_no_face = True
    if 'train' in datasetname:
        skip_multiple_faces = True
        save_no_face = False

    loc = args.dataset_path
    save_loc = loc + "_faces"
    generate_folder(save_loc)

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=256, margin=64, keep_all=True, min_face_size=80)

    files = list(sorted(glob.glob(os.path.join(loc, "*"))))
    labels = pd.read_csv(os.path.join(os.path.dirname(loc), os.path.basename(loc) + ".csv"))
    category = pd.read_csv(os.path.join(os.path.dirname(loc), "category.csv"))
    new_labels = pd.DataFrame(columns=labels.columns[1:])

    transform = transforms.Compose([resize(1024, 1024, keys=('x',)), padding(1024, 1024, keys=('x',), mode='constant')])
    ds = ImageDataset(files, files, n_classes=len(category), onehot=False, transform=transform)
    ds = DataLoader(ds, batch_size=args.batch_size, num_workers=4, shuffle=False)

    face_count_hist = [0] * 40
    pbar = tqdm(ds)
    for sample in pbar:
        imgs = sample['x']
        pbar.set_postfix_str(f"{face_count_hist[:5]}")
        imgs_crop, face_probs = extract_faces(imgs * 255, mtcnn)
        for i in range(len(imgs)):
            filename = os.path.basename(sample['y'][i])
            label = labels[labels["File Name"] == filename].Category.item()
            probs = face_probs[i]
            faces = imgs_crop[i]
            _image = imgs[i] * 255
            if faces is None:
                faces, probs = try_bad_image(_image, mtcnn)
            if faces is None:
                face_count_hist[0] += 1
                if save_no_face:
                    new_labels = save_face(new_labels, _image.to(torch.uint8).numpy(), label, filename, 0, save_loc)
                continue
            faces = faces[probs > 0.99]
            probs = probs[probs > 0.99]
            if len(faces) == 0:
                faces, probs = try_bad_image(_image, mtcnn)
            if len(faces) == 0:
                face_count_hist[0] += 1
                if save_no_face:
                    new_labels = save_face(new_labels, _image.to(torch.uint8).numpy(), label, filename, 0, save_loc)
                continue

            face_count_hist[len(faces)] += 1

            if len(probs) > 1 and skip_multiple_faces:
                continue
            for j in range(len(faces)):
                face = einops.rearrange(faces[j], 'c h w -> h w c')
                face = ((face + 1) / 2 * 255).to(torch.uint8).numpy()
                img = _image.to(torch.uint8).numpy()
                prob = probs[j]

                new_labels = save_face(new_labels, face, label, filename, j, save_loc)

                # new_filename = f"{filename.split('.')[0]}_{j:03d}.{filename.split('.')[1]}"
                # new_labels.loc[len(new_labels)] = [new_filename, label]
                # plt.imsave(os.path.join(save_loc, new_filename), face)

    new_labels.to_csv(save_loc + ".csv")
    plt.bar(np.arange(len(face_count_hist)), face_count_hist)
    plt.savefig(loc + '_count.png')
