import os
from datetime import datetime
from os.path import dirname, join
import torch
import numpy as np
from matplotlib import pyplot as plt

from utils.loss import loss_parse
from utils.utils import AverageMeter, get_basename


class Logger():
    def __init__(self, args, save_dir=None):
        self.args = args
        if save_dir is None:
            now = datetime.now() if 'time' not in vars(args) else args.time
            now = now.strftime("%Y-%m-%d %H.%M.%S")
            file_path = join(args.save_dir, f"{now} {args.name}({get_basename(self.args.dataset_path)})", 'log.txt')
            self.save_dir = dirname(file_path)
        else:
            self.save_dir = save_dir
            args.save_dir = save_dir
            file_path = join(args.save_dir, 'log.txt')
        print(f"Logger initialized. Saving to {self.save_dir}")
        self.check_dir(file_path)
        self.logger = open(file_path, 'a+')

        # variable register
        self.register_dict = {}

        # loss meters
        self.loss_meters = {}

        # tensorboard
        if self.args.phase != 'infer':
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.save_dir)
        else:
            self.writer = None

    def record_args(self):
        self('recording parameters ...')
        args_file = open(join(self.save_dir, 'args'), 'w')
        for key, value in vars(self.args).items():
            self(f'{key}: {value}', timestamp=False)
            args_file.write(f'{key}: {value}\n')
            args_file.flush()
        args_file.close()

    def record_kwargs(self, kwargs):
        self('recording kwargs...')
        args_file = open(join(self.save_dir, 'kwargs'), 'w')
        for key, value in vars(kwargs).items():
            self(f'{key}: {value}', timestamp=False)
            args_file.write(f'{key}: {value}\n')
            args_file.flush()
        args_file.close()

    def check_dir(self, file_path):
        dir = dirname(file_path)
        os.makedirs(dir, exist_ok=True)

    def __call__(self, *args, verbose=False, prefix='', timestamp=True):
        if timestamp:
            now = datetime.now()
            now = now.strftime("%Y/%m/%d, %H:%M:%S - ")
        else:
            now = ''
        info = prefix + now
        for msg in args:
            if not isinstance(msg, str):
                msg = str(msg)
            info += msg + '\n'
        self.logger.write(info)
        if verbose:
            print(info, end='')
        self.logger.flush()

    def __del__(self):
        self.logger.close()

    # register values for each epoch, such as loss, PSNR etc.
    def __register(self, name, epoch, value):
        if name in self.register_dict:
            self.register_dict[name][epoch] = value
            if value > self.register_dict[name]['max']:
                self.register_dict[name]['max'] = value
            if value < self.register_dict[name]['min']:
                self.register_dict[name]['min'] = value
        else:
            self.register_dict[name] = {}
            self.register_dict[name][epoch] = value
            self.register_dict[name]['max'] = value
            self.register_dict[name]['min'] = value

    def report(self, items, state, epoch):
        # items - [['MSE', 'min'], ['PSNR', 'max'] ... ]
        msg = '[{}] '.format(state.lower())
        state = '_' + state.lower()
        for i in range(len(items)):
            item, best = items[i]
            msg += f'{item} : {self.register_dict[item + state][epoch]:.4f} ({best} {self.register_dict[item + state][best]:.4f})'
            if i < len(items) - 1:
                msg += ', '
        self(msg, timestamp=False)

    def save_model_params(self, params, model_name):
        assert type(params) == dict
        par_path = join(self.save_dir, f"{model_name}")
        np.savez(par_path, **params)

    def create_loss_meters(self, name_loss_dict):
        for name, loss in name_loss_dict.items():
            self.loss_meters[name] = {}
            _, losses_names = loss_parse(loss)
            losses_names += ['all']
            for key in losses_names:
                self.loss_meters[name][key] = AverageMeter()

    def update_losses(self, name, losses, n_batch):
        if 'all' not in losses.keys():
            for k in list(losses.keys()):
                if 'all' not in losses.keys():
                    losses['all'] = torch.zeros_like(losses[k])
                losses['all'] += losses[k]

        for key in losses.keys():
            if key not in self.loss_meters[name].keys():
                self.loss_meters[name][key] = AverageMeter()
            self.loss_meters[name][key].update(losses[key].detach().item(), n_batch)

    def register_losses(self, name, phase, epoch):
        self.__register(f'Total_{phase}', epoch, self.loss_meters[name]['all'].avg)
        if self.writer is None:
            return
        for key in self.loss_meters[name].keys():
            self.writer.add_scalar(key + f'_loss_{phase}', self.loss_meters[name][key].avg, epoch)

    def plot_losses(self, losses, is_log=True):
        """

        @param losses: of shape (2, n_epochs) first array = val, second array = train
        @param is_log: use loglog?
        @return: figure
        """
        fig = plt.figure()
        # if len(losses[0]) < 2:
        #     print("Skipping plot because not enough data points to plot.")
        #     return fig
        if is_log:
            plt.loglog(losses[1], losses[0], '-bD', label='val')
            plt.loglog(losses[3], losses[2], '-ro', label='train')
        else:
            plt.plot(losses[1], losses[0], '-bD', label='val')
            plt.plot(losses[3], losses[2], '-ro', label='train')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(join(self.save_dir, "losses.png"))
        return fig
    # def is_best(self, epoch):
    #     item = self.register_dict[self.args.loss + '_valid']
    #     return item[epoch] == item['min']
    #
    # def save(self, state, filename='checkpoint.pth.tar'):
    #     path = join(self.save_dir, filename)
    #     torch.save(state, path)
    #     if self.is_best(state['epoch']):
    #         copy_path = join(self.save_dir, 'model_best.pth.tar')
    #         shutil.copy(path, copy_path)
