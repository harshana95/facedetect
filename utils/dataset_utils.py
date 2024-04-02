import cv2
import einops
import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms
import torch.nn.functional as F


class resize:
    def __init__(self, h, w, keys, keep_aspect_ratio=True):
        self.h = h
        self.w = w
        self.keys = keys
        self.kar = keep_aspect_ratio
        self.trans = transforms.Resize(size=(h, w),antialias=True)

    def _resize(self, x):
        w, h = self.w, self.h

        if self.kar:
            if type(x) == np.ndarray:
                xh, xw = x.shape[0], x.shape[1]
            else:
                xh, xw = x.shape[1], x.shape[2]
            if w>h:
                w = self.w
                h = int(xh*w/xw)
            else:
                h = self.h
                w = int(xw*h/xh)
            self.trans = transforms.Resize(size=(h, w),antialias=True)

        if type(x) == np.ndarray:
            resized = cv2.resize(x, (w, h))
        else:
            resized = self.trans(x)
        return resized

    def __call__(self, sample):

        if type(sample) == dict:
            for key in self.keys:
                sample[key] = self._resize(sample[key])
        else:
            sample = self._resize(sample)
        return sample


class divisible_by:
    def __init__(self, n, keys, method='crop'):
        self.n = n
        self.keys = keys
        self.method = 0 if method == 'crop' else 1

    @staticmethod
    def _crop(x, n):
        h, w = x.shape[-2:]
        if n > 1:
            H = h - h % n
            W = w - w % n
            return x[..., :H, :W]
        return x

    @staticmethod
    def _resize(x, n):
        c, h, w = x.shape
        H = (h // n) * n
        W = (w // n) * n
        resized = cv2.resize(x.transpose([1, 2, 0]), (W, H))
        return resized.transpose([2, 0, 1])

    def __call__(self, sample):
        if type(sample) == dict:
            for key in self.keys:
                if self.method == 0:
                    sample[key] = self._crop(sample[key], self.n)
                else:
                    sample[key] = self._resize(sample[key], self.n)
        else:
            if self.method == 0:
                sample = self._crop(sample, self.n)
            else:
                sample = self._resize(sample, self.n)
        return sample


class crop2d:
    def __init__(self, crop_indices=None, best=True, keys=()):
        self.best = best
        self.active_keys = keys
        self.ci = crop_indices
        if best and crop_indices is not None:
            print("Crop indices will be ignored and best square will be cropped")

    def crop(self, image):
        # sample shape (..., h, w)
        if self.best:
            h, w = image.shape[-2], image.shape[-1]
            ch, cw = h // 2, w // 2
            r = min(ch, cw)
            image = image[..., ch - r:ch + r, cw - r:cw + r]
        else:
            image = image[..., self.ci[0]:self.ci[1], self.ci[2]:self.ci[3]]
        return image

    def __call__(self, sample):
        if type(sample) == dict:
            for key in self.active_keys:
                sample[key] = self.crop(sample[key])
        else:
            sample = self.crop(sample)
        return sample


class fix_image_shape:
    def __init__(self, keys):
        self.keys = keys

    def fix(self, image):
        if len(image.shape) == 4:
            raise Exception("too many dimensions. expected 3 dimensions")
            # if image.shape[1] == 3 or image.shape[1] == 1:
            #     pass
            # elif image.shape[-1] == 3 or image.shape[-1] == 1:
            #     image = image.transpose([0, 3, 1, 2])
            # else:
            #     raise Exception(f"unknown image shape {image.shape}")
        elif len(image.shape) == 3:
            if image.shape[-1] == 3 or image.shape[-1] == 1:
                image = image.transpose([2, 0, 1])

        elif len(image.shape) == 2:
            image = image[None, :]
        else:
            # not an image
            return image

        return image

    def __call__(self, sample):
        if type(sample) == dict:
            for key in self.keys:
                sample[key] = self.fix(sample[key])
        else:
            sample = self.fix(sample)
        return sample


class grayscale:
    def convert2grayscale(self, image):
        # Convert to grayscale
        # Input should be a tensor
        # (N C H W) or (C H W) or (H W)
        if len(image.shape) == 4:
            raise Exception("Too many dimensions")

        elif len(image.shape) == 3:
            r, g, b = image.unbind(dim=-3)
            l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(image.dtype)
            l_img = l_img.unsqueeze(dim=-3)
            # image = image.mean(0, keepdims=True)
        elif len(image.shape) == 2:
            l_img = image[None, :]
        else:
            return image
            # raise Exception("grayscale failed")

        return l_img

    def __call__(self, sample):
        if type(sample) == dict:
            for key in sample.keys():
                sample[key] = self.convert2grayscale(sample[key])
        else:
            sample = self.convert2grayscale(sample)
        return sample


class add_gaussian_noise:
    def __init__(self, max_sigma=0.02, sigma_dc=0.005, mu=0, keys=(), astype=torch.float32):
        self.max_sigma = max_sigma  # abit much maybe 0.04 best0.04+0.01
        self.sigma_dc = sigma_dc
        self.mu = mu
        self.active_keys = keys
        self.astype = astype

    def add_noise(self, sample):
        if type(sample) != torch.Tensor:
            sample = torch.from_numpy(sample)

        shape = sample.shape
        sigma = np.random.rand() * self.max_sigma + self.sigma_dc
        g_noise = np.random.normal(self.mu, sigma, np.prod(shape)).reshape(shape)
        g_noise = torch.from_numpy(g_noise).to(sample.device)
        ret = sample + g_noise.to(self.astype)
        # ret = ret / torch.max(ret)
        # ret = torch.maximum(ret, torch.zeros_like(ret))
        ret = torch.clamp(ret, 0, 1)
        return ret, torch.tensor([sigma])

    def __call__(self, sample):
        if type(sample) == dict:
            for key in self.active_keys:
                sample[key], sample[f'{key}_gauss_param'] = self.add_noise(sample[key])
        else:
            sample = self.add_noise(sample)
        return sample


class add_poisson_noise:
    def __init__(self, peak=1000, peak_dc=50, keys=(), astype=torch.float32):
        super().__init__()
        self.PEAK = peak  # np.random.rand(1) * 1000 + 50
        self.PEAK_DC = peak_dc
        self.astype = astype
        self.active_keys = keys

    def add_noise(self, sample):
        if type(sample) != torch.Tensor:
            sample = torch.from_numpy(sample)
        peak = np.random.rand() * self.PEAK + self.PEAK_DC
        if peak < 0:
            return sample, torch.tensor([0])
        p_noise = torch.poisson(torch.clamp(sample, min=1e-6) * peak)  # poisson cannot take negative
        p_noise = p_noise.to(sample.device)
        # ret = p_noise
        ret = sample + (p_noise.to(self.astype) / peak)
        # ret = ret / torch.max(ret)
        # ret = torch.maximum(ret, torch.zeros_like(ret))
        ret = torch.clamp(ret, 0, 1)
        return ret, torch.tensor([peak])

    def __call__(self, sample):
        if type(sample) == dict:
            for key in self.active_keys:
                sample[key], sample[f'{key}_poisson_param'] = self.add_noise(sample[key])
        else:
            sample = self.add_noise(sample)
        return sample


class to_tensor:
    def __call__(self, sample, astype=torch.float32):
        if type(sample) == dict:
            for key in sample.keys():
                if type(sample[key]) != torch.Tensor:
                    if type(sample[key]) == np.ndarray:
                        sample[key] = torch.from_numpy(sample[key]).to(astype)
        else:
            sample = torch.from_numpy(sample).to(astype)
        return sample


class padding:
    def __init__(self, h, w, keys, mode='reflect'):
        super().__init__()
        self.h = h
        self.w = w
        self.keys = keys
        self.mode = mode

    def __call__(self, sample, astype=torch.float32):

        if type(sample) == dict:
            for key in self.keys:
                sample[key] = einops.rearrange(sample[key], 'h w c -> c h w')
                sample[key] = crop_arr(sample[key], self.h, self.w, mode=self.mode)
                sample[key] = einops.rearrange(sample[key], 'c h w -> h w c')
        else:
            sample = einops.rearrange(sample, 'h w c -> c h w')
            sample = crop_arr(sample, self.h, self.w, mode=self.mode)
            sample = einops.rearrange(sample, 'c h w -> h w c')

        return sample


class normalize:
    def __call__(self, sample, astype=torch.float32):
        if type(sample) == dict:
            for key in sample.keys():
                sample[key] = (sample[key] / torch.max(sample[key])).to(astype)
        else:
            sample = (sample / torch.max(sample)).to(astype)
        return sample


class augment:

    def __init__(self,  keys, horizontal_flip=True, resize_crop=True, image_shape=None, rotate=True):
        self.keys = keys
        self.ops = []
        if horizontal_flip:
            self.ops.append(transforms.RandomHorizontalFlip())
        if resize_crop:
            assert image_shape is not None
            self.ops.append(transforms.RandomResizedCrop(image_shape, antialias=True))
        if rotate:
            self.ops.append(transforms.RandomRotation(90))

    def aug(self, image):
        for op in self.ops:
            image = op(image)
        return image

    def __call__(self, sample, astype=torch.float32):
        if type(sample) == dict:
            for key in self.keys:
                sample[key] = self.aug(sample[key]).to(astype)
        else:
            sample = self.aug(sample).to(astype)
        return sample


def crop_arr(arr, h, w, mode='constant'):
    hw, ww = arr.shape[-2:]
    do_pad = False
    if type(arr) == torch.Tensor:
        pad = [0, 0, 0, 0]
    else:
        pad = [[0, 0]] * (len(arr.shape))
    if h < hw:
        crop_height = min(h, hw)
        top = hw // 2 - crop_height // 2
        arr = arr[..., top:top + crop_height, :]
    elif h > hw:
        do_pad = True
        if type(arr) == torch.Tensor:
            pad[-2] = int(np.ceil((h - hw) / 2))
            pad[-1] = int(np.floor((h - hw) / 2))
        else:
            pad[-2] = [int(np.ceil((h - hw) / 2)), int(np.floor((h - hw) / 2))]
    if w < ww:
        crop_width = min(w, ww)
        left = ww // 2 - crop_width // 2
        arr = arr[..., :, left:left + crop_width]
    elif w > ww:
        do_pad = True
        if type(arr) == torch.Tensor:
            pad[0] = int(np.ceil((w - ww) / 2))
            pad[1] = int(np.floor((w - ww) / 2))
        else:
            pad[-1] = [int(np.ceil((w - ww) / 2)), int(np.floor((w - ww) / 2))]
    if do_pad:
        if type(arr) == torch.Tensor:
            arr = torch.nn.functional.pad(arr, pad, mode=mode)
        else:
            arr = np.pad(arr, pad, mode=mode)
    return arr


def crop_from_center(sample: dict, ref_key: str, crop_keys: list):
    """
    Crop crop_key in sample wrt to ref_key from the center. h, w should be the last 2 dimensions
    @param sample: dict with arrays
    @param ref_key: reference array
    @param crop_keys: keys that should be cropped
    @return:
    """
    ref = sample[ref_key]
    hx, wx = ref.shape[-2:]

    for crop_key in crop_keys:
        crp = sample[crop_key]
        sample[crop_key] = crop_arr(crp, hx, wx)
    return sample
