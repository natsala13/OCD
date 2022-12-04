import torch
import torch.nn as nn
import numpy as np
from nerf_utils.nerf import cumprod_exclusive, get_minibatches, get_ray_bundle, positional_encoding
from nerf_utils.tiny_nerf import VeryTinyNerfModel
from torchvision.datasets import mnist
from torchvision import transforms
import Lenet5
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from copy import deepcopy

from utils_OCD import ConfigWrapper

from spice.model.sim2sem import Sim2Sem
from spice.data.build_dataset import build_dataset
from fixmatch.models.nets.wrn import WideResNet
from fixmatch.datasets.ssl_dataset_robust import SSL_Dataset
from fixmatch.datasets.data_utils import get_data_loader

WIDERESNET_DEPTH = 28
WIDERESNET_WIDDEN = 2
WIDERESNET_LEAKY_SLOPE = 0.1
WIDERESNET_DROPRATE = 0.0


def build_model_for_cifar10(config: ConfigWrapper, args, device):
    print('* Building model')
    model = WideResNet(depth=WIDERESNET_DEPTH,
                       num_classes=10,
                       widen_factor=WIDERESNET_WIDDEN,
                       bn_momentum=0.1,
                       leaky_slope=WIDERESNET_LEAKY_SLOPE,
                       dropRate=WIDERESNET_DROPRATE)

    print('* Loading checkpoint to model')
    checkpoint = torch.load(config['spice']['model']['pretrained'], map_location=torch.device('cpu'))
    state_dict = {k.replace('module.', '', 1): checkpoint['eval_model'][k] for k in checkpoint['eval_model']}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    _train_dset = SSL_Dataset(name='cifar10',
                              train=True,
                              data_dir="./datasets/cifar10",
                              label_file=None,
                              all=False,
                              unlabeled=False)

    _eval_dset = SSL_Dataset(name='cifar10',
                             train=False,
                             data_dir="./datasets/cifar10",
                             label_file=None,
                             all=False,
                             unlabeled=False)

    train_dset = _train_dset.get_dset()
    eval_dset = _eval_dset.get_dset()

    # train_ds, _ = torch.utils.data.random_split(train_dset, [20000, 30000])
    # test_ds, _ = torch.utils.data.random_split(eval_dset, [1000, 9000])

    train_loader = get_data_loader(train_dset, 1, num_workers=1)
    eval_loader = get_data_loader(eval_dset, 1, num_workers=1)

    return train_loader, eval_loader, model


def spice_model_build(config: ConfigWrapper, args, device):
    if args.weight:
        config['spice_model']['pretrained'] = args.weight

    # create model
    model = Sim2Sem(**config['spice_model'])

    print('* Loading checkpoint to model')
    checkpoint = torch.load(config['spice']['model']['pretrained'], map_location=torch.device('cpu'))
    state_dict = {k.replace('module.', '', 1): checkpoint['eval_model'][k] for k in checkpoint['eval_model']}
    model.load_state_dict(state_dict)
    model = model.to(device)

    # *************** Dataset ****************
    print('* Loading train data')
    train_dataset = build_dataset(config['spice']['data_train'])

    # TODO: batch size = 1?, num workers = 4?
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)

    print('* Loading test data')
    dataset_val = build_dataset(config['spice']['data_test'])
    test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1)

    return train_loader, test_loader, model


def wrapper_dataset(config: ConfigWrapper, args, device):
    if args.datatype == 'tinynerf':

        data = np.load(args.data_train_path)
        images = data["images"]
        # Camera extrinsics (poses)
        tform_cam2world = data["poses"]
        tform_cam2world = torch.from_numpy(tform_cam2world).to(device)
        # Focal length (intrinsics)
        focal_length = data["focal"]
        focal_length = torch.from_numpy(focal_length).to(device)

        # Height and width of each image
        height, width = images.shape[1:3]

        # Near and far clipping thresholds for depth values.
        near_thresh = 2.0
        far_thresh = 6.0

        # Hold one image out (for test).
        testimg, testpose = images[101], tform_cam2world[101]
        testimg = torch.from_numpy(testimg).to(device)

        # Map images to device
        images = torch.from_numpy(images[:100, ..., :3]).to(device)
        num_encoding_functions = 10
        # Specify encoding function.
        encode = positional_encoding
        # Number of depth samples along each ray.
        depth_samples_per_ray = 32
        model = VeryTinyNerfModel(num_encoding_functions=num_encoding_functions)
        # Chunksize (Note: this isn't batchsize in the conventional sense. This only
        # specifies the number of rays to be queried in one go. Backprop still happens
        # only after all rays from the current "bundle" are queried and rendered).
        # Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory (when using 8
        # samples per ray).
        chunksize = 4096
        batch = {}
        batch['height'] = height
        batch['width'] = width
        batch['focal_length'] = focal_length
        batch['testpose'] = testpose
        batch['near_thresh'] = near_thresh
        batch['far_thresh'] = far_thresh
        batch['depth_samples_per_ray'] = depth_samples_per_ray
        batch['encode'] = encode
        batch['get_minibatches'] = get_minibatches
        batch['chunksize'] = chunksize
        batch['num_encoding_functions'] = num_encoding_functions
        train_ds, test_ds = [], []
        for img, tfrom in zip(images, tform_cam2world):
            batch['input'] = tfrom
            batch['output'] = img
            train_ds.append(deepcopy(batch))
        batch['input'] = testpose
        batch['output'] = testimg
        test_ds = [batch]
    elif args.datatype == 'mnist':
        model = Lenet5.NetOriginal()
        train_transform = transforms.Compose(
            [
                transforms.ToTensor()
            ])
        train_dataset = mnist.MNIST(
            "\data\mnist", train=True, download=True, transform=ToTensor())
        test_dataset = mnist.MNIST(
            "\data\mnist", train=False, download=True, transform=ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1)
        train_ds, test_ds = [], []
        for idx, data in enumerate(train_loader):
            train_x, train_label = data[0], data[1]
            train_x = train_x[:, 0, :, :].unsqueeze(1)
            batch = {'input': train_x, 'output': train_label}
            train_ds.append(deepcopy(batch))
        for idx, data in enumerate(test_loader):
            train_x, train_label = data[0], data[1]
            train_x = train_x[:, 0, :, :].unsqueeze(1)
            batch = {'input': train_x, 'output': train_label}
            test_ds.append(deepcopy(batch))
    elif args.datatype == 'cifar10':
        train_ds, test_ds, model = build_model_for_cifar10(config, args, device)
    else:
        "implement on your own"
        pass
    return train_ds, test_ds, model
