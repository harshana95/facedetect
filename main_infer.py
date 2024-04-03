import argparse
import glob
from os.path import join, dirname

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from dataset.BasicDataset import ImageDataset
from trainers.trainer import Trainer
from utils.dataset_utils import resize, fix_image_shape
from utils.default_args import add_default_args, parse_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser, 'face_detect', 64, phase='infer', datasetname='test_faces')
    parser.add_argument('--use_best_face', action='store_true', help='Finetune')
    args = parse_args(parser)

    trainer = Trainer(args)
    trainer.initialize_model()
    trainer.model = trainer.model.cuda()

    labels = pd.read_csv(join(dirname(args.dataset_path), 'category.csv'))
    n_classes = len(labels)

    y = pd.read_csv(args.dataset_path + '.csv')
    y['Path'] = [join(args.dataset_path, i) for i in y['File Name']]
    y['Label'] = [0 for i in y["Path"]]
    y["Id"] = [i.split("_")[0] for i in y["File Name"]]
    y["Id"] = y["Id"].astype(int)
    y = y.sort_values(by=['Id'])

    ans = pd.DataFrame(columns=["Id", "Category"])

    unique_id = sorted(y["Id"].unique())

    transform = transforms.Compose([resize(args.image_size_h, args.image_size_w, keys=('x',)),
                                    fix_image_shape(keys=('x',))])
    with torch.no_grad():
        for _id in tqdm(unique_id):
            idx = y["Id"] == _id
            _labels = y['Label'][idx]  # dummy labels
            _f_paths = sorted(y['Path'][idx])  # file paths
            if args.use_best_face:
                _f_paths = _f_paths[:1]
            face_ds = ImageDataset(_f_paths, _labels, n_classes, transform=transform, onehot=False)
            n_faces = len(face_ds)

            best_score = -1e10
            best_label = -1
            for sample in face_ds:
                x = sample[trainer.x_key]
                x = torch.tensor(x[None]).cuda()

                x_hat, label = trainer.model(x)
                label = label.cpu().numpy()[0]

                score = label.max()
                arg_label = label.argmax()

                if score > best_score:
                    best_score = score
                    best_label = arg_label

                #plt.subplot(131), plt.imshow(x[0].numpy().transpose([1, 2, 0])), plt.title(_id)
                #plt.subplot(132), plt.imshow(x_hat[0].numpy().transpose([1, 2, 0])), plt.title(f"{labels[labels.index == arg_label]['Category'].item()}")
                #plt.subplot(133), plt.plot(label)
                #plt.show()
            if best_label == -1:
                print("No faces", _id)
                best_label = 0
            best_label_name = labels[labels.index == best_label]["Category"].item()
            ans.loc[len(ans)] = [_id, best_label_name]
            if _id%100 == 0:
                ans["Id"] = ans["Id"].astype(int)
                ans = ans.sort_values(by=['Id'])
                ans.to_csv(args.load_model_path+"Solution.csv", index=False)

    ans["Id"] = ans["Id"].astype(int)
    ans = ans.sort_values(by=['Id'])
    ans.to_csv(args.load_model_path+"Solution.csv", index=False)


