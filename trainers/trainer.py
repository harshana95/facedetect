import glob
import os
import re
import shutil
import timeit
import matplotlib

import cv2
import pandas as pd
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import scipy
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset.BasicDataset import ImageDataset
from modelzoo.AutoEncoder import *
from modelzoo.FaceModel import *
from utils.dataset_utils import *
from utils.loss_utils import eval_metrics_folder
from utils.utils import generate_folder, dict_add, hstack_images

from os.path import join, dirname, basename

from einops import einops
from utils.logger import Logger
from utils.loss import Loss

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import torch
from accelerate import Accelerator
import matplotlib as mpl
from tqdm import tqdm

mpl.rc('image', cmap='inferno')


class Trainer:
    _phase_train = 0
    _phase_test = 1
    _phase_val = 2
    _phase_infer = 3

    def __init__(self, args):
        self.dataset_test, self.dataset_train, self.dataset_val = None, None, None
        self.model, self.optimizer = None, None
        self.dataset_paths = []
        self.onthefly = False
        self.sched = None
        self.x_key, self.y_key = 'x', 'y'
        self.n_classes = -1
        self.args = args
        self.patchify = self.args.patch_size > 0
        np.random.seed(9)
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.best_acc = -1
        self.batch_size = args.batch_size
        print(f"Batch size {self.batch_size}")

        # check if continuing or loading model
        updatable_args = ['save_dir', 'current_epoch']
        non_updatable_args = ['phase', 'dataset_path', 'load_model_path', 'gt_data_path', 'meas_data_path',
                              'val_gt_data_path', 'val_meas_data_path', 'metrics', 'max_sigma', 'dc_sigma',
                              'test_name', 'patch_size', 'stride'
                              ]
        if 'load_model_path' in vars(self.args) and len(self.args.load_model_path) > 0:
            self.load_folder = os.path.dirname(self.args.load_model_path)
            # try loading args
            print("Loading args")
            with open(join(self.load_folder, 'args')) as f:
                while f.readable():
                    line = f.readline()
                    if len(line) == 0:
                        break

                    key, val = line.split(': ')

                    if key not in updatable_args and args.phase == 'train':
                        continue
                    if key in non_updatable_args:
                        continue
                    val = val.strip()
                    try:
                        val = float(val)
                        if val.is_integer():
                            val = int(val)
                    except:
                        pass
                    if val == "True":
                        val = True
                    elif val == "False":
                        val = False
                    if not self.args.__contains__(key):
                        print(f"Added: ({key}) - {val}")
                        self.args.__setattr__(key, val)
                    if self.args.__getattribute__(key) != val:
                        print(f"Updated: ({key}) {self.args.__getattribute__(key)} -> {val}")
                        self.args.__setattr__(key, val)
        else:
            self.load_folder = None

        # create logger
        self.logger = Logger(self.args, save_dir=self.load_folder)
        if self.load_folder is None:
            self.logger.record_args()
        self.results_dir = join(self.logger.save_dir, 'results')
        self.inference_results_dir = join(self.logger.save_dir, 'inference_results', self.args.test_name)
        try:
            shutil.rmtree(self.inference_results_dir)
        except Exception:
            pass
        generate_folder(self.results_dir)
        generate_folder(self.inference_results_dir)

        self.criterion = Loss(args, self.args.loss_image).to(self.device)
        self.criterion_label = Loss(args, self.args.loss_label).to(self.device)

        # setup dataset paths
        if args.dataset_path != '':
            self.dataset_paths.append(args.dataset_path)
        if args.dataset_paths is not None:
            for ds_path in args.dataset_paths:
                self.dataset_paths.append(ds_path)
        print(f"*************** Dataset paths {self.dataset_paths}")
        self.initialize_dataset(onthefly=self.onthefly, dataset_path=self.dataset_paths[0])

        # setup comet.ml
        experiment_name = os.path.basename(self.logger.save_dir)
        if self.args.phase == 'infer':
            self.experiment = Experiment(
                api_key="ytMuLR3yqWuCA2lDrsgcHEmMr",
                disabled=True
            )
        else:
            self.experiment = Experiment(
                api_key="ytMuLR3yqWuCA2lDrsgcHEmMr",
                project_name=self.args.name,
                workspace="harshana95",
            )
            for pyfile in glob.glob(f"{os.path.dirname(self.args.cwd)}/**/*.py", recursive=True):
                self.experiment.log_code(pyfile)
        self.args.experiment_name = experiment_name
        self.experiment.log_parameters(self.args.__dict__)
        self.experiment.set_name(experiment_name)

    def initialize_dataset(self, onthefly=True, dataset_path=None, refresh=False):
        """
        Initialize dataset_train, dataset_val, dataset_test, dataset_patched_infer datasets
        @param dataset_path:
        @param onthefly:
        @return:
        """
        if dataset_path is None:
            dataset_path = self.args.dataset_path

        y = pd.read_csv(dataset_path + '.csv')
        labels = pd.read_csv(join(dirname(dataset_path), 'category.csv'))
        self.n_classes = len(labels)

        y['Path'] = [join(dataset_path, i) for i in y['File Name']]
        y['Label'] = [labels.Category[labels.Category == i].index.tolist()[0] for i in y.Category]

        if not refresh:
            x_train, x_test, y_train, y_test = train_test_split(y['Path'], y['Label'], test_size=self.args.test_ratio,
                                                                shuffle=False)
            x_test = x_test
            y_test = y_test
            self.x_train = x_train
            self.y_train = y_train

            self.x_test = x_test
            self.y_test = y_test

        x_train, x_val, y_train, y_val = train_test_split(self.x_train, self.y_train, test_size=self.args.val_ratio)
        x_val = x_val[:100]
        y_val = y_val[:100]
        # make labels noisy
        n = np.random.rand(len(y_train)) < self.args.label_noise
        y_train[n] = np.random.randint(0, 100, sum(n))

        h, w = self.args.image_size_h, self.args.image_size_w
        one2r2 = 1/(2*(2**0.5))
        transform = transforms.Compose([
            fix_image_shape(keys=('x',)),
            to_tensor(),
            resize(h, w, keys=('x',)),
            augment(keys=('x',), horizontal_flip=True, resize_crop=False, rotate=True),
            cropresize(h, w, crop_indices=[int(h*(0.5-one2r2)), int(h*(0.5+one2r2)), int(w*(0.5-one2r2)), int(w*(0.5+one2r2))], random=True, keys=('x',)),
        ])
        dataset_train = ImageDataset(x_train, y_train, self.n_classes, transform=transform)
        dataset_val = ImageDataset(x_val, y_val, self.n_classes, transform=transform)
        dataset_test = ImageDataset(self.x_test, self.y_test, self.n_classes, transform=transform)

        n_workers = self.args.n_workers
        # setting dataset and split into train and validation

        self.dataset_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=n_workers,
                                        drop_last=False)
        self.dataset_val = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=n_workers,
                                      drop_last=False)
        self.dataset_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=n_workers)

    def check_model_speed(self):
        iters = len(self.dataset_train)
        wait = 1
        warmup = 4
        active = 3
        repeat = 1
        n = (wait + warmup + active) * repeat
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t = []

        self.logger.create_loss_meters({"model": self.args.loss_image})

        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.logger.save_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
        ) as prof:
            for i_batch, sample_batched in enumerate(self.dataset_train):
                # for _ in range(n):
                start.record()
                self.optimizer.zero_grad()
                out, label, loss = self.model_forward(sample_batched, self.criterion, 0)

                loss['all'].backward()
                self.optimizer.step()
                self.sched.step(i_batch / iters)

                end.record()
                torch.cuda.synchronize()
                t.append(start.elapsed_time(end))

                # add loss to the meter
                self.logger.update_losses('model', loss, len(sample_batched[self.x_key]))

                prof.step()
                if i_batch > n:
                    break
        print(f"Average time after {n} iterations {np.mean(t) / 1000}. Max {np.max(t) / 1000} Min {np.min(t) / 1000}")
        print(f"Saved profiler to {self.logger.save_dir}")

    def initialize_model(self, *args, **kwargs):
        if self.args.model == 'model1':
            ae = AutoEncoder(n_channel_in=3, n_channel_out=3)
            self.model = FaceModel((self.args.image_size_h, self.args.image_size_w), n_classes=self.n_classes, ae=ae)
        elif self.args.model == 'model2':
            ae = AutoEncoder2(n_channel_in=3, n_channel_out=3)
            self.model = FaceModel((self.args.image_size_h, self.args.image_size_w), n_classes=self.n_classes, ae=ae)
        elif self.args.model == 'model1_small':
            ae = AutoEncoder(n_channel_in=3, n_channel_out=3)
            self.model = FaceModelSmall((self.args.image_size_h, self.args.image_size_w), n_classes=self.n_classes,
                                        ae=ae)
        elif self.args.model == 'model2_small':
            ae = AutoEncoder2(n_channel_in=3, n_channel_out=3)
            self.model = FaceModelSmall((self.args.image_size_h, self.args.image_size_w), n_classes=self.n_classes,
                                        ae=ae)
        elif self.args.model == 'simple':
            self.model = Simple()
        elif self.args.model == 'simple_medium':
            self.model = SimpleMedium()
        elif self.args.model == 'simple_medium2':
            self.model = SimpleMedium2()
        elif self.args.model == 'simple_small':
            self.model = SimpleSmall()
        elif self.args.model == 'inc':
            self.model = InceptionV1()
        elif self.args.model == 'inc2':
            self.model = InceptionV2()
        elif self.args.model == 'inc3':
            self.model = InceptionV3()
        elif self.args.model == 'inc4':
            self.model = InceptionV4()
        elif self.args.model == 'inc5':
            self.model = InceptionV5()
        else:
            raise Exception()
        if len(self.args.load_model_path) > 0:
            print(f"Loading model from {self.args.load_model_path}")
            self.model.load_state_dict(torch.load(self.args.load_model_path))

    def save_model(self, filename):
        torch.save(self.model.state_dict(), join(self.logger.save_dir, f'{filename}.pt'))
        # self.experiment.log_model(self.args.experiment_name, join(self.logger.save_dir, f'{filename}.pt'))

    def model_forward(self, sample, criterion, phase, optimizer=None):
        """
        @param optimizer: give optimizer if you want to backprop
        @param sample: data
        @param criterion:
        @param phase: 0-train, 1-test, 2-val, 3-infer
        @return:
        """
        if optimizer:
            for param in self.model.parameters():
                param.grad = None
        # with torch.cuda.amp.autocast():
        x = sample[self.x_key].to(self.device)
        x_hat, label = self.model(x)
        loss1 = criterion(x_hat, x)
        loss2 = self.criterion_label(label, sample[self.y_key])
        loss_sum = dict_add(loss1, loss2, 1.)
        if optimizer:
            loss_sum['all'].backward()
            optimizer.step()

        return x_hat, label, loss_sum

    def train(self, debug=False):
        self.experiment.set_model_graph(self.model, overwrite=False)

        criterion = self.criterion
        if self.optimizer is None:
            print("Creating default optimizer Adam, default scheduler CosineAnnealingWarmRestarts")
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
            self.sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=2,
                                                                              eta_min=1e-6)

        self.model, self.optimizer, self.dataset_train = self.accelerator.prepare(self.model, self.optimizer,
                                                                                  self.dataset_train)

        total_param = sum(p.numel() for p in self.model.parameters()) / 1000000
        self.experiment.log_parameter("Total parameters (M)", total_param)
        print(f"Total parameters in {self.args.model} model {total_param:.2f}M")
        if debug:
            print("initial tests to check the speed of dataloader and network")
            self.check_model_speed()

        print("Started training")
        self.model.train()
        _tmp_losses_to_plot = [[], [], [], []]
        total_epochs = self.args.current_epoch + self.args.epochs

        for epoch in range(self.args.current_epoch, total_epochs):
            self.args.current_epoch = epoch  # this will change self.logger.args as well
            self.logger.create_loss_meters({"model": criterion.loss_str})

            if epoch % self.args.validate_every == 0:
                with torch.no_grad():
                    if self.dataset_test is not None:
                        self.test_validate(phase='test', dataset=self.dataset_test, save_images=True, save_model=True)
                    self.test_validate(phase='val', dataset=self.dataset_val, save_images_first_only=False,
                                       save_model=False)
                    _tmp_losses_to_plot[0].append(self.logger.loss_meters['model']['all'].avg)
                    _tmp_losses_to_plot[1].append(epoch)
            if epoch % self.args.refresh_dataset_every == 0:
                print(f"Refreshing datasets")
                self.initialize_dataset(onthefly=self.onthefly, dataset_path=self.dataset_paths[0], refresh=True)
                self.dataset_train = self.accelerator.prepare(self.dataset_train)

            iters = len(self.dataset_train)
            pbar = tqdm(self.dataset_train)
            mean_acc = 0
            mean_acc_n = 0
            for i_batch, sample_batched in enumerate(pbar):
                # optimize the model
                out, label, loss = self.model_forward(sample_batched, criterion, self._phase_train, self.optimizer)
                self.sched.step(epoch + (i_batch / iters))

                # add loss to the meter
                self.logger.update_losses('model', loss, out.shape[0] * out.shape[1] if self.patchify else len(out))

                # check accuracy
                _x = torch.argmax(sample_batched[self.y_key], dim=1, keepdim=True)
                _y = torch.argmax(label, dim=1, keepdim=True)
                acc = ((_x == _y).float().mean())
                loss['Accuracy'] = acc
                mean_acc += acc * len(_x)
                mean_acc_n += len(_x)

                # update current status
                tmp = f"Epoch: {epoch}/{total_epochs} loss:{loss['all'].item():.2f} acc:{mean_acc / mean_acc_n:.2f} lr:{self.sched.get_last_lr()}"
                pbar.set_postfix_str(tmp)
            pbar.close()
            self.experiment.log_metric(f"Accuracy (train)", mean_acc / mean_acc_n, epoch=self.args.current_epoch)
            # save losses in tensorboard
            _tmp_losses_to_plot[2].append(self.logger.loss_meters['model']['all'].avg)
            _tmp_losses_to_plot[3].append(epoch)
            self.logger.plot_losses(_tmp_losses_to_plot)
            self.logger.register_losses('model', 'train', epoch)
            # self.logger.writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
            # scheduler.step()
            self.logger.record_args()
            if self.args.save_checkponts:
                self.save_model("model_last")

            # change dataset
            with torch.no_grad():
                if len(self.dataset_paths) > 1:
                    self.initialize_dataset(self.onthefly, self.dataset_paths[(epoch + 1) % len(self.dataset_paths)])

        # do inference on test dataset
        self.test_validate('infer', self.dataset_test, save_images=True)
        log_model(self.experiment, self.model, self.args.name)
        print("Training complete!")

    def test_validate(self, phase='test', dataset=None, save_images=False, save_images_first_only=False,
                      return_data=False, save_model=False):
        _valid = ['test', 'val', 'infer']
        assert phase in _valid
        _phase = _valid.index(phase) + 1

        if dataset is None:
            dataset = self.dataset_test

        self.model, dataset = self.accelerator.prepare(self.model, dataset)
        self.model.eval()

        if phase == 'infer' or phase == 'test' or phase == 'train':
            loc = str(self.inference_results_dir)
        else:
            loc = str(self.results_dir)

        self.logger.create_loss_meters({"model": self.criterion.loss_str})
        with torch.no_grad():
            idx = 0
            pbar = tqdm(dataset)
            mean_acc = 0
            mean_acc_n = 0
            for i_batch, sample_batched in enumerate(pbar):
                out, label, loss = self.model_forward(sample_batched, self.criterion, _phase)
                self.logger.update_losses('model', loss, out.shape[0] * out.shape[1] if self.patchify else len(out))

                # get acc
                _x = torch.argmax(sample_batched[self.y_key], dim=1, keepdim=True)
                _y = torch.argmax(label, dim=1, keepdim=True)
                acc = ((_x == _y).float().mean())
                loss['Accuracy'] = acc
                mean_acc += acc * len(_x)
                mean_acc_n += len(_x)

                out_np = einops.rearrange(out.detach().cpu().numpy(), "n c h w -> n h w c")
                gt_np = einops.rearrange(sample_batched[self.x_key].detach().cpu().numpy(), "n c h w -> n h w c")

                if save_images:
                    if gt_np.shape[-1] == 1:
                        gt_np = einops.repeat(gt_np, "n h w 1 -> n h w C", C=3)
                    if out_np.shape[-1] == 1:
                        out_np = einops.repeat(out_np, "n h w 1 -> n h w C", C=3)
                    if self.args.save_renormalized:
                        gt_np = (gt_np - gt_np.min((1, 2, 3), keepdims=True)) / (
                                gt_np.max((1, 2, 3), keepdims=True) - gt_np.min((1, 2, 3), keepdims=True))
                        out_np = (out_np - out_np.min((1, 2, 3), keepdims=True)) / (
                                out_np.max((1, 2, 3), keepdims=True) - out_np.min((1, 2, 3), keepdims=True))
                    for i in range(gt_np.shape[0]):
                        idx += 1
                        name = f"{phase}_{idx}_{self.args.current_epoch:03d}.png"
                        to_save = hstack_images([gt_np[i].squeeze(), out_np[i].squeeze()], 0, 1)
                        self.experiment.log_image((np.clip(out_np[i].squeeze(), 0, 1) * 255).astype(np.uint8),
                                                  name=f'{idx}_{phase}', overwrite=False,
                                                  step=self.args.current_epoch)
                        fig = plt.figure()
                        if to_save.shape[-1] == 1 or len(to_save.shape) == 2:
                            plt.imshow(to_save, cmap='gray')
                        else:
                            plt.imshow(to_save)
                        plt.title(f"GT {_x[i, 0]} Pred {_y[i, 0]}")
                        plt.savefig(join(loc, name))
                        plt.close(fig)

                # update current status
                avg_loss = self.logger.loss_meters['model']['all'].avg
                tmp = f"Phase: {phase} loss:{avg_loss:.2f} acc:{mean_acc / mean_acc_n:.2f}"
                pbar.set_postfix_str(tmp)
            pbar.close()
            self.experiment.log_metric(f"Accuracy ({phase})", mean_acc / mean_acc_n, epoch=self.args.current_epoch)
            # save losses in tensorboard
            self.logger.register_losses('model', phase, self.args.current_epoch)

        if mean_acc / mean_acc_n > self.best_acc:
            self.best_acc = mean_acc / mean_acc_n

            # save checkpoint
            if self.args.save_checkponts and save_model:
                print("Saving best model")
                self.save_model(f"model_best_{self.args.current_epoch}")
        self.model.train()
