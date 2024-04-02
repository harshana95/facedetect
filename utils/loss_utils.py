import glob
import os
from os.path import join

import cv2
import einops
import numpy as np
import torch

from utils.dataset_utils import crop_arr
from utils.utils import dict_add


def eval_metrics(x, y, cr):
    pr = cv2.imread(x)
    gt = cv2.imread(y)
    pr = np.array(pr)[None, :, :, :] / 255.0
    gt = np.array(gt)[None, :, :, :] / 255.0
    pr = einops.rearrange(pr, 'b h w c -> b c h w')
    gt = einops.rearrange(gt, 'b h w c -> b c h w')
    h = min(pr.shape[-2], gt.shape[-2])
    w = min(pr.shape[-1], gt.shape[-1])
    pr = crop_arr(pr, h, w)
    gt = crop_arr(gt, h, w)
    pr = torch.Tensor(pr).cuda()
    gt = torch.Tensor(gt).cuda()
    return cr(pr, gt)


def eval_metrics_folder(folder, cr):
    pr_images = sorted(glob.glob(join(folder, "reconstructed", "*")))
    gt_images = sorted(glob.glob(join(folder, "gt", "*")))
    n = len(pr_images)
    results = {}

    metrics_f = open(join(folder, 'metrics.txt'), 'w+')
    metrics_f.write(cr.loss_str + "\n")
    for i in range(n):
        loss = eval_metrics(gt_images[i], pr_images[i], cr)
        results = dict_add(results, loss)

        _s = f"{os.path.basename(pr_images[i])}\t"
        for key in loss.keys():
            _l = loss[key].detach().item()
            _s += str(_l) + "\t"
        metrics_f.write(_s[:-1] + "\n")

    _s1, _s2 = "", ""
    for key in results.keys():
        results[key] /= n
        _s1 += str(key) + "\t"
        _s2 += str(results[key].detach().item()) + "\t"
    metrics_f.write(_s1[:-1] + "\n")
    metrics_f.write(_s2[:-1] + "\n")
    metrics_f.close()
    print("average")
    print(results)
