import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

from utils.dataset_utils import normalize


class Conv(nn.Module):
    def __init__(self, input_channels, n_feats, kernel_size, stride=1, padding=0, bias=True, bn=False, act=False):
        super(Conv, self).__init__()
        m = [nn.Conv2d(input_channels, n_feats, kernel_size, stride, padding, bias=bias)]
        if bn: m.append(nn.BatchNorm2d(n_feats))
        if act: m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)


class Deconv(nn.Module):
    def __init__(self, input_channels, n_feats, kernel_size, stride=2, padding=0, output_padding=0, bias=True,
                 act=False):
        super(Deconv, self).__init__()
        m = [nn.ConvTranspose2d(input_channels, n_feats, kernel_size, stride=stride, padding=padding,
                                output_padding=output_padding, bias=bias)]
        if act: m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down='conv', kernel_size=2, stride=2, groups=1):
        super(DownBlock, self).__init__()
        if down == 'maxpool':
            self.down1 = nn.MaxPool2d(kernel_size=kernel_size)
        elif down == 'avgpool':
            self.down1 = nn.AvgPool2d(kernel_size=kernel_size)
        elif down == 'conv':
            self.down1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups)
            self.down1.weight.data = 0.01 * self.down1.weight.data + 0.25
            self.down1.bias.data = 0.01 * self.down1.bias.data + 0

    def forward(self, x):
        return self.down1(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up='tconv', kernel_size=2, stride=2, groups=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if up == 'bilinear' or up == 'nearest':
            scale_factor = kwargs.get("scale_factor")
            self.up1 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=scale_factor)
        elif up == 'tconv':
            self.up1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                          groups=groups)
            self.up1.weight.data = 0.01 * self.up1.weight.data + 0.25
            self.up1.bias.data = 0.01 * self.up1.bias.data + 0

    def forward(self, x):
        return self.up1(x)


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=True,
                 dropout=False, norm='none', residual=True, activation='gelu',
                 transpose=False):
        super(ConvBlock, self).__init__()
        self.dropout = dropout
        self.residual = residual
        self.activation = activation
        self.transpose = transpose

        if self.dropout:
            self.dropout1 = nn.Dropout2d(p=0.05)
            self.dropout2 = nn.Dropout2d(p=0.05)

        self.norm1 = None
        self.norm2 = None
        if norm == 'batch':
            self.norm1 = nn.BatchNorm2d(outc)
            self.norm2 = nn.BatchNorm2d(outc)
        elif norm == 'instance':
            self.norm1 = nn.InstanceNorm2d(outc, affine=True)
            self.norm2 = nn.InstanceNorm2d(outc, affine=True)
        elif norm == 'mixed':
            self.norm1 = nn.BatchNorm2d(outc, affine=True)
            self.norm2 = nn.InstanceNorm2d(outc, affine=True)

        if self.transpose:
            self.conv1 = nn.ConvTranspose2d(inc, outc, kernel_size=kernel_size, padding=padding, bias=bias)
            self.conv2 = nn.ConvTranspose2d(outc, outc, kernel_size=kernel_size, padding=padding, bias=bias)
        else:
            self.conv1 = nn.Conv2d(inc, outc, kernel_size=kernel_size, padding=padding, bias=bias)
            self.conv2 = nn.Conv2d(outc, outc, kernel_size=kernel_size, padding=padding, bias=bias)

        if self.activation == 'relu':
            self.actfun1 = nn.ReLU()
            self.actfun2 = nn.ReLU()
        elif self.activation == 'leakyrelu':
            self.actfun1 = nn.LeakyReLU()
            self.actfun2 = nn.LeakyReLU()
        elif self.activation == 'elu':
            self.actfun1 = nn.ELU()
            self.actfun2 = nn.ELU()
        elif self.activation == 'selu':
            self.actfun1 = nn.SELU()
            self.actfun2 = nn.SELU()
        elif self.activation == 'gelu':
            self.actfun1 = nn.GELU()
            self.actfun2 = nn.GELU()

    def forward(self, x):
        ox = x
        x = self.conv1(x)
        if self.dropout:
            x = self.dropout1(x)
        if self.norm1:
            x = self.norm1(x)
        x = self.actfun1(x)
        x = self.conv2(x)
        if self.dropout:
            x = self.dropout2(x)
        if self.norm2:
            x = self.norm2(x)
        if self.residual:
            x[:, 0:min(ox.shape[1], x.shape[1]), :, :] += ox[:, 0:min(ox.shape[1], x.shape[1]), :, :]
        x = self.actfun2(x)
        # print("shapes: x:%s ox:%s " % (x.shape,ox.shape))
        return x


class ResBlock(nn.Module):
    def __init__(self, body, res_scale=1):
        super(ResBlock, self).__init__()
        self.body = body
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class CropBlock(nn.Module):
    def __init__(self, crop_indices, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crop_indices = crop_indices

    def forward(self, x):
        return x[..., self.crop_indices[0]:self.crop_indices[1], self.crop_indices[2]:self.crop_indices[3]]


class RDB_Conv(nn.Module):
    def __init__(self, in_channels, grow_rate, kernel=3):
        super(RDB_Conv, self).__init__()
        self.conv = nn.Sequential(*[
            nn.Conv2d(in_channels, grow_rate, kernel, padding=(kernel - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, in_channels, n_layers, grow_rate, kernel=3):
        super(RDB, self).__init__()
        G0 = in_channels
        G = grow_rate
        N = n_layers

        conv = []
        for n in range(N):
            conv.append(RDB_Conv(G0 + n * G, G, kernel))
        self.conv = nn.Sequential(*conv)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + N * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.conv(x)) + x


class FeatureBlock(nn.Module):
    def __init__(self, n_feats1, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        class _ResBlock(nn.Module):
            def __init__(self, conv, n_feat, kernel_size, padding=0, bias=True, bn=False, act=nn.ReLU(True),
                         res_scale=1):
                super(_ResBlock, self).__init__()
                m = []
                for i in range(2):
                    m.append(conv(n_feat, n_feat, kernel_size, padding=padding, bias=bias))
                    if bn: m.append(nn.BatchNorm2d(n_feat))
                    if i == 0: m.append(act)

                self.body = nn.Sequential(*m)
                self.res_scale = res_scale

            def forward(self, x):
                res = self.body(x).mul(self.res_scale)
                res += x

                return res

        self.featureBlock = nn.Sequential(Conv(3, n_feats1, kernel_size, padding=2, act=True),
                                          _ResBlock(Conv, n_feats1, kernel_size, padding=2),
                                          _ResBlock(Conv, n_feats1, kernel_size, padding=2),
                                          _ResBlock(Conv, n_feats1, kernel_size, padding=2))

    def forward(self, x):
        return self.featureBlock(x)


class Ensemble(nn.Module):
    def __init__(self, models_list, transforms=None):
        super(Ensemble, self).__init__()
        if transforms is None:
            print("No transforms defined. Usually add 'normalize' after each model.")
            transforms = torchvision.transforms.Compose([])
        self.model_list = nn.ModuleList(models_list)
        self.transforms = transforms


class SerialEnsemble(Ensemble):
    """
        x -> [model1] -> [model2] -> out
    """

    def forward(self, x):
        for model in self.model_list:
            x = model(x)
            if self.transforms:
                x = self.transforms(x)
        return x


class ParallelEnsemble(Ensemble):
    """
        |--[ model 1 ] -> y1
       x                  |-> out = concat(y1,y2)
       |--[ model 2 ]-> y2
    """

    def __init__(self, models_list, transforms=None, concat_dim=0):
        super().__init__(models_list, transforms)
        self.concat_dim = concat_dim

    def forward(self, x):
        outs = []
        for model in self.model_list:
            out = model(x)
            if self.transforms:
                out = self.transforms(out)
            outs.append(out)
        return torch.concat(outs, dim=self.concat_dim)


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


# copied from https://github.com/developer0hye/PyTorch-Deformable-Convolution-v2/blob/main/dcn.py
class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        # op = (n - (k * d - 1) + 2p / s)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=(self.padding, self.padding),
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=(self.dilation, self.dilation))
        return x


# copied from https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=-3)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=-3)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
