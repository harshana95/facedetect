import argparse
import os
from os.path import join


def add_default_args(parser, name, batch_size=1, phase='train', datasetname='train_small_faces'):
    if os.name == 'nt':
        name = 'test_' + name
        dataset_path = f'F:\\Datasets\\Face\\{datasetname}'
        data_percent = 100
        # dataset_path = 'C:\\Users\\harsh\\Datasets\\flickr30k_images_PCA_file_z10000_cartesian_x1'
    elif os.name == 'posix' and os.path.isdir('/depot'):
        dataset_path = f"/scratch/gilbreth/wweligam/FaceDetect/{datasetname}"
        data_percent = 100
    else:
        from PyQt5.QtCore import QLibraryInfo
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        name = 'test_' + name
        dataset_path = f"/home/harshana/Documents/Datasets/{datasetname}"
        data_percent = 100

    # experiment parameters
    parser.add_argument('--name', default=name)
    parser.add_argument('--save_dir', default=os.path.join('experiments', name))
    parser.add_argument('--dataset_path', default=dataset_path)
    parser.add_argument('--dataset_paths', nargs='+', help='Set of datasets to use')
    parser.add_argument('--test_data_path', default='./dataset/test/')
    parser.add_argument('--test_name', default='test', help='inference test name')
    parser.add_argument('--data_percent', type=float, default=data_percent, help='data % to use')
    parser.add_argument('--load_model_path', default='')
    parser.add_argument('--cwd', type=str, default=os.path.dirname(__file__), help='cwd')
    parser.add_argument('--show_plots', action='store_true', help='Show plots')
    parser.add_argument('--save_renormalized', type=int, default=1, help='Re-Normalize images before saving')
    parser.add_argument('--model', type=str, default='inc')

    # image/psf parameters
    parser.add_argument('--n_colors', type=int, default=3, help='image color channels')
    parser.add_argument('--image_size_h', type=int, default=256, help='image height')
    parser.add_argument('--image_size_w', type=int, default=256, help='image width')
    parser.add_argument('--psf_size', default=96, type=int, help='psf size when needed. might not use in some methods')
    parser.add_argument('--patch_size', default=0, type=int, help='patch size to patchify images. 0 means no patching')
    parser.add_argument('--stride', default=-1, type=int, help='stride of patching. should be < patch_size')
    parser.add_argument('--psf_sampling', default='basis', type=str, choices=['basis', 'uniform', 'random'])
    parser.add_argument('--n_psf_sampling', default=100, type=int, help='number of psf samples to get')
    parser.add_argument('--padding', type=int, default=32)
    parser.add_argument('--divisible', type=int, default=16, help='Make image divible by this number. Useful for unets')
    parser.add_argument('--deconv_nsr_weight', type=float, default=1., help='Deconvolution NSR weight')

    # training parameters
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='batch size')
    parser.add_argument('--loss_image', type=str, default='1*L1', help='loss function, e.g.1*MSE|1e-4*Perceptual')
    parser.add_argument('--loss_label', type=str, default='1*CrossEntropyLoss', help='metrics')
    parser.add_argument('--save_checkponts', default=True)
    parser.add_argument('--current_epoch', type=int, default=0)
    parser.add_argument('--validate_every', type=int, default=10)
    parser.add_argument('--refresh_dataset_every', type=int, default=50)
    parser.add_argument('--phase', default=phase, choices=['train', 'infer'])
    parser.add_argument('--n_workers', default=4, type=int, help='number of workers in the dataloader')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--finetune', action='store_true', help='Finetune')
    parser.add_argument('--finetune_path', type=str, help='Finetune')
    parser.add_argument('--test_ratio', type=float, default=0.002)
    parser.add_argument('--val_ratio', type=float, default=0.02)

    # noise parameters
    parser.add_argument('--max_sigma', type=float, default=0.05, help='max variance of the noise')
    parser.add_argument('--dc_sigma', type=float, default=0.001, help='added constant variance of the noise')
    parser.add_argument('--mu_sigma', type=float, default=0., help='mean of the noise')
    parser.add_argument('--peak_poisson', type=float, default=-1, help='peak poisson noise')  # 1000
    parser.add_argument('--peak_dc_poisson', type=float, default=0, help='peak dc poisson')  # 50
    parser.add_argument('--label_noise', type=float, default=.00)

    return parser


def parse_args(parser):
    args, kwargs = parser.parse_known_args()
    for i, arg in enumerate(kwargs):
        if arg.startswith(("-", "--")):
            _type = str
            if i < len(kwargs) - 1:
                try:
                    tmp = float(kwargs[i + 1])
                    _type = float
                    if float(kwargs[i + 1]) == int(kwargs[i + 1]):
                        _type = int
                except:
                    pass
            parser.add_argument(arg, type=_type)
    args = parser.parse_args()
    assert args.patch_size > args.stride

    print(args)
    return args
