# -*- coding: utf-8 -*-
import numpy as np
import scipy
from scipy import fftpack
import torch
import torch.nn.functional as F

from modelzoo.Blocks import MedianPool2d

# import utils_image as util

'''
modified by Kai Zhang (github: https://github.com/cszn)
03/03/2019
'''


def get_uperleft_denominator(img, kernel):
    ker_f = convert_psf2otf(kernel, img.size())  # discrete fourier transform of kernel
    nsr = wiener_filter_para(img)
    # print(nsr)
    denominator = inv_fft_kernel_est(ker_f, nsr)  #
    img1 = img.cuda()
    # numerator = torch.rfft(img1, 3, onesided=False)
    numerator = torch.fft.fft2(img1, dim=(-2, -1))
    deblur = deconv(denominator, numerator)
    return deblur


def wiener_filter_para(_input_blur):
    median_filter = MedianPool2d(kernel_size=3, padding=1)(_input_blur)
    diff = median_filter - _input_blur
    num = (diff.shape[2] * diff.shape[2])
    mean_n = torch.sum(diff, (2, 3)).view(-1, 1, 1, 1) / num
    var_n = torch.sum((diff - mean_n) * (diff - mean_n), (2, 3)) / (num - 1)
    mean_input = torch.sum(_input_blur, (2, 3)).view(-1, 1, 1, 1) / num
    var_s2 = (torch.sum((_input_blur - mean_input) * (_input_blur - mean_input), (2, 3)) / (num - 1)) ** (0.5)
    NSR = var_n / var_s2 * 8.0 / 3.0 / 10.0
    NSR = NSR.view(-1, 1, 1, 1)
    return NSR


def inv_fft_kernel_est(ker_f, NSR):
    inv_denominator = ker_f[:, :, :, :].real * ker_f[:, :, :, :].real \
                      + ker_f[:, :, :, :].imag * ker_f[:, :, :, :].imag + NSR
    # pseudo inverse kernel in flourier domain.
    inv_ker_f = torch.zeros_like(ker_f)
    inv_ker_f[:, :, :, :].real = ker_f[:, :, :, :].real / inv_denominator
    inv_ker_f[:, :, :, :].imag = -ker_f[:, :, :, :].imag / inv_denominator
    return inv_ker_f


def deconv(inv_ker_f, fft_input_blur):
    # delement-wise multiplication.
    deblur_f = torch.zeros_like(inv_ker_f).cuda()
    deblur_f[:, :, :, :].real = inv_ker_f[:, :, :, :].real * fft_input_blur[:, :, :, :].real \
                                - inv_ker_f[:, :, :, :].imag * fft_input_blur[:, :, :, :].imag
    deblur_f[:, :, :, :].imag = inv_ker_f[:, :, :, :].real * fft_input_blur[:, :, :, :].imag \
                                + inv_ker_f[:, :, :, :].imag * fft_input_blur[:, :, :, :].real
    # deblur = torch.irfft(deblur_f, 3, onesided=False)
    deblur = torch.fft.ifft2(deblur_f, dim=(-2, -1))
    return deblur


# --------------------------------
# --------------------------------
def wiener_filter_nsr(_input_blur):
    """

    @param _input_blur: (N C H W)
    @return: (N C 1 1)
    """
    n, c, h, w = _input_blur.shape
    num = h * w
    median_filter = MedianPool2d(kernel_size=3, padding=1)(_input_blur)
    diff = median_filter - _input_blur

    mean_n = torch.sum(diff, (-2, -1)).view(n, c, 1, 1) / num
    var_n = torch.sum((diff - mean_n) * (diff - mean_n), (-2, -1)) / (num - 1)
    mean_input = torch.sum(_input_blur, (-2, -1)).view(n, c, 1, 1) / num
    var_s2 = (torch.sum((_input_blur - mean_input) * (_input_blur - mean_input), (-2, -1)) / (num - 1)) ** (0.5)
    NSR = var_n / var_s2 * 8.0 / 3.0 / 10.0
    NSR = NSR.view(n, c, 1, 1)
    return NSR


# --------------------------------
# --------------------------------
def convert_psf2otf(ker, size):
    """
        Convert point-spread function to optical transfer function.
        otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
        point-spread function (PSF) array and creates the optical transfer
        function (OTF) array that is not influenced by the PSF off-centering.

        Args:
            psf: (C1 C2 N h w)
            shape: (H W)

        Returns:
            otf: (C1 C2 N H W 2)
        """
    shape = list(ker.shape[:-2])
    shape += list(size)
    psf = torch.zeros(shape).to(ker.device)
    # if ker.shape[-2] % 2 == 0:
    #     ker = F.pad(ker, (0, 1, 0, 1), "constant", 0)

    # h_psf, w_psf = ker.shape[-2:]
    # h, w = size
    # if h_psf < h or w_psf < w:  # Pad psf
    #     padding_psf = [int(np.ceil(h_psf / 2)), int(np.floor(h_psf / 2)),
    #                    int(np.ceil(w_psf / 2)), int(np.floor(w_psf / 2))]
    #     ker = F.pad(ker, padding_psf)
    # if h_psf > h or w_psf > w:  # Crop psf
    #     top, left = h_psf // 2 - h // 2, w_psf // 2 - w // 2
    #     ker = ker[:, top:top + h, left:left + w]
    # xs, ys = size[0] // 2 - ker.shape[-2] // 2, size[1] // 2 - ker.shape[-1] // 2
    # xe = xs + ker.shape[-2]
    # ye = ys + ker.shape[-1]
    # psf[:, :, :, xs:xe, ys:ye] = ker

    # circularly shift
    # put PSF quadrants to the 4 corners of the 2d plane
    h1, w1 = ker.shape[-2]//2, ker.shape[-1]//2
    h2, w2 = ker.shape[-2]-h1, ker.shape[-1]-w1
    psf[..., :h1, :w1] = ker[..., -h1:, -w1:]
    psf[..., :h1, -w2:] = ker[..., -h1:, :w2]
    psf[..., -h2:, :w1] = ker[..., :h2, -w1:]
    psf[..., -h2:, -w2:] = ker[..., :h2, :w2]

    # center = ker.shape[-2] // 2 + 1
    # psf[..., :center, :center] = ker[..., (center - 1):, (center - 1):]
    # psf[..., :center, -(center - 1):] = ker[..., (center - 1):, :(center - 1)]
    # psf[..., -(center - 1):, :center] = ker[..., :(center - 1), (center - 1):]
    # psf[..., -(center - 1):, -(center - 1):] = ker[..., :(center - 1), :(center - 1)]

    # compute the otf
    otf = torch.fft.fft2(psf, dim=(-2, -1))
    return otf


def postprocess(*images, rgb_range):
    def _postprocess(img):
        pixel_range = 255 / rgb_range
        return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

    return [_postprocess(img) for img in images]


# ======================================================================================================================

# psf2otf copied/modified from https://github.com/aboucaud/pypher/blob/master/pypher/pypher.py
def psf2otf(psf, shape=None):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.array`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if type(shape) == type(None):
        shape = psf.shape
    shape = np.array(shape)

    if np.all(psf == 0):
        # return np.zeros_like(psf)
        return np.zeros(shape)

    if len(psf.shape) == 1:
        psf = psf.reshape((1, psf.shape[0]))

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf


# otf2psf: not sure where I got this one from. Maybe translated from Octave source code or whatever. It's just math.
def otf2psf(otf, outsize=None):
    insize = np.array(otf.shape)
    psf = np.fft.ifftn(otf, axes=(0, 1))
    for axis, axis_size in enumerate(insize):
        psf = np.roll(psf, np.floor(axis_size / 2).astype(int), axis=axis)
    if type(outsize) != type(None):
        insize = np.array(otf.shape)
        outsize = np.array(outsize)
        n = max(np.size(outsize), np.size(insize))
        # outsize = postpad(outsize(:), n, 1);
        # insize = postpad(insize(:) , n, 1);
        colvec_out = outsize.flatten().reshape((np.size(outsize), 1))
        colvec_in = insize.flatten().reshape((np.size(insize), 1))
        outsize = np.pad(colvec_out, ((0, max(0, n - np.size(colvec_out))), (0, 0)), mode="constant")
        insize = np.pad(colvec_in, ((0, max(0, n - np.size(colvec_in))), (0, 0)), mode="constant")

        pad = (insize - outsize) / 2
        if np.any(pad < 0):
            print("otf2psf error: OUTSIZE must be smaller than or equal than OTF size")
        prepad = np.floor(pad)
        postpad = np.ceil(pad)
        dims_start = prepad.astype(int)
        dims_end = (insize - postpad).astype(int)
        for i in range(len(dims_start.shape)):
            psf = np.take(psf, range(dims_start[i][0], dims_end[i][0]), axis=i)
    n_ops = np.sum(otf.size * np.log2(otf.shape))
    psf = np.real_if_close(psf, tol=n_ops)
    return psf


def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)
    if np.alltrue(imshape == shape):
        return image
    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")
    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")
    pad_img = np.zeros(shape, dtype=image.dtype)
    idx, idy = np.indices(imshape)
    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)
    pad_img[idx + offx, idy + offy] = image
    return pad_img


'''
Reducing boundary artifacts
'''


def opt_fft_size(n):
    '''
    Kai Zhang (github: https://github.com/cszn)
    03/03/2019
    #  opt_fft_size.m
    # compute an optimal data length for Fourier transforms
    # written by Sunghyun Cho (sodomau@postech.ac.kr)
    # persistent opt_fft_size_LUT;
    '''

    LUT_size = 2048
    # print("generate opt_fft_size_LUT")
    opt_fft_size_LUT = np.zeros(LUT_size)

    e2 = 1
    while e2 <= LUT_size:
        e3 = e2
        while e3 <= LUT_size:
            e5 = e3
            while e5 <= LUT_size:
                e7 = e5
                while e7 <= LUT_size:
                    if e7 <= LUT_size:
                        opt_fft_size_LUT[e7 - 1] = e7
                    if e7 * 11 <= LUT_size:
                        opt_fft_size_LUT[e7 * 11 - 1] = e7 * 11
                    if e7 * 13 <= LUT_size:
                        opt_fft_size_LUT[e7 * 13 - 1] = e7 * 13
                    e7 = e7 * 7
                e5 = e5 * 5
            e3 = e3 * 3
        e2 = e2 * 2

    nn = 0
    for i in range(LUT_size, 0, -1):
        if opt_fft_size_LUT[i - 1] != 0:
            nn = i - 1
        else:
            opt_fft_size_LUT[i - 1] = nn + 1

    m = np.zeros(len(n))
    for c in range(len(n)):
        nn = n[c]
        if nn <= LUT_size:
            m[c] = opt_fft_size_LUT[nn - 1]
        else:
            m[c] = -1
    return m


def wrap_boundary_liu(img, img_size):
    """
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    if img.ndim == 2:
        ret = wrap_boundary(img, img_size)
    elif img.ndim == 3:
        ret = [wrap_boundary(img[:, :, i], img_size) for i in range(3)]
        ret = np.stack(ret, 2)
    return ret


def wrap_boundary(img, img_size):
    """
    python code from:
    https://github.com/ys-koshelev/nla_deblur/blob/90fe0ab98c26c791dcbdf231fe6f938fca80e2a0/boundaries.py
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    (H, W) = np.shape(img)
    H_w = int(img_size[0]) - H
    W_w = int(img_size[1]) - W

    # ret = np.zeros((img_size[0], img_size[1]));
    alpha = 1
    HG = img[:, :]

    r_A = np.zeros((alpha * 2 + H_w, W))
    r_A[:alpha, :] = HG[-alpha:, :]
    r_A[-alpha:, :] = HG[:alpha, :]
    a = np.arange(H_w) / (H_w - 1)
    # r_A(alpha+1:end-alpha, 1) = (1-a)*r_A(alpha,1) + a*r_A(end-alpha+1,1)
    r_A[alpha:-alpha, 0] = (1 - a) * r_A[alpha - 1, 0] + a * r_A[-alpha, 0]
    # r_A(alpha+1:end-alpha, end) = (1-a)*r_A(alpha,end) + a*r_A(end-alpha+1,end)
    r_A[alpha:-alpha, -1] = (1 - a) * r_A[alpha - 1, -1] + a * r_A[-alpha, -1]

    r_B = np.zeros((H, alpha * 2 + W_w))
    r_B[:, :alpha] = HG[:, -alpha:]
    r_B[:, -alpha:] = HG[:, :alpha]
    a = np.arange(W_w) / (W_w - 1)
    r_B[0, alpha:-alpha] = (1 - a) * r_B[0, alpha - 1] + a * r_B[0, -alpha]
    r_B[-1, alpha:-alpha] = (1 - a) * r_B[-1, alpha - 1] + a * r_B[-1, -alpha]

    if alpha == 1:
        A2 = solve_min_laplacian(r_A[alpha - 1:, :])
        B2 = solve_min_laplacian(r_B[:, alpha - 1:])
        r_A[alpha - 1:, :] = A2
        r_B[:, alpha - 1:] = B2
    else:
        A2 = solve_min_laplacian(r_A[alpha - 1:-alpha + 1, :])
        r_A[alpha - 1:-alpha + 1, :] = A2
        B2 = solve_min_laplacian(r_B[:, alpha - 1:-alpha + 1])
        r_B[:, alpha - 1:-alpha + 1] = B2
    A = r_A
    B = r_B

    r_C = np.zeros((alpha * 2 + H_w, alpha * 2 + W_w))
    r_C[:alpha, :] = B[-alpha:, :]
    r_C[-alpha:, :] = B[:alpha, :]
    r_C[:, :alpha] = A[:, -alpha:]
    r_C[:, -alpha:] = A[:, :alpha]

    if alpha == 1:
        C2 = C2 = solve_min_laplacian(r_C[alpha - 1:, alpha - 1:])
        r_C[alpha - 1:, alpha - 1:] = C2
    else:
        C2 = solve_min_laplacian(r_C[alpha - 1:-alpha + 1, alpha - 1:-alpha + 1])
        r_C[alpha - 1:-alpha + 1, alpha - 1:-alpha + 1] = C2
    C = r_C
    # return C
    A = A[alpha - 1:-alpha - 1, :]
    B = B[:, alpha:-alpha]
    C = C[alpha:-alpha, alpha:-alpha]
    ret = np.vstack((np.hstack((img, B)), np.hstack((A, C))))
    return ret


def solve_min_laplacian(boundary_image):
    (H, W) = np.shape(boundary_image)

    # Laplacian
    f = np.zeros((H, W))
    # boundary image contains image intensities at boundaries
    boundary_image[1:-1, 1:-1] = 0
    j = np.arange(2, H) - 1
    k = np.arange(2, W) - 1
    f_bp = np.zeros((H, W))
    f_bp[np.ix_(j, k)] = -4 * boundary_image[np.ix_(j, k)] + boundary_image[np.ix_(j, k + 1)] + boundary_image[
        np.ix_(j, k - 1)] + boundary_image[np.ix_(j - 1, k)] + boundary_image[np.ix_(j + 1, k)]

    del (j, k)
    f1 = f - f_bp  # subtract boundary points contribution
    del (f_bp, f)

    # DST Sine Transform algo starts here
    f2 = f1[1:-1, 1:-1]
    del (f1)

    # compute sine tranform
    if f2.shape[1] == 1:
        tt = fftpack.dst(f2, type=1, axis=0) / 2
    else:
        tt = fftpack.dst(f2, type=1) / 2

    if tt.shape[0] == 1:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt), type=1, axis=0) / 2)
    else:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt), type=1) / 2)
    del (f2)

    # compute Eigen Values
    [x, y] = np.meshgrid(np.arange(1, W - 1), np.arange(1, H - 1))
    denom = (2 * np.cos(np.pi * x / (W - 1)) - 2) + (2 * np.cos(np.pi * y / (H - 1)) - 2)

    # divide
    f3 = f2sin / denom
    del (f2sin, x, y)

    # compute Inverse Sine Transform
    if f3.shape[0] == 1:
        tt = fftpack.idst(f3 * 2, type=1, axis=1) / (2 * (f3.shape[1] + 1))
    else:
        tt = fftpack.idst(f3 * 2, type=1, axis=0) / (2 * (f3.shape[0] + 1))
    del (f3)
    if tt.shape[1] == 1:
        img_tt = np.transpose(fftpack.idst(np.transpose(tt) * 2, type=1) / (2 * (tt.shape[0] + 1)))
    else:
        img_tt = np.transpose(fftpack.idst(np.transpose(tt) * 2, type=1, axis=0) / (2 * (tt.shape[1] + 1)))
    del (tt)

    # put solution in inner points; outer points obtained from boundary image
    img_direct = boundary_image
    img_direct[1:-1, 1:-1] = 0
    img_direct[1:-1, 1:-1] = img_tt
    return img_direct
