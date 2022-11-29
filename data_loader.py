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

import attrdict
from pathlib import Path
from utils_OCD import ConfigWrapper

from spice.model.sim2sem import Sim2Sem
from spice.data.build_dataset import build_dataset


def build_model_for_cifar10(config: ConfigWrapper, args, device):
    if args.weight:
        config['spice']['model']['pretrained'] = args.weight

    # **************** Model ****************
    model = Sim2Sem(**config['spice']['model'])
    print(model)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # model.eval()  # model.train(False)

    # TODO: Load checkpoint
    checkpoint = torch.load(config['spice']['model']['pretrained'], map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    # if cfg.model.pretrained is not None:
    #     load_model_weights(model, cfg.model.pretrained, cfg.model.model_type)
    # ****************************************

    # TODO: DO I need it?
    # optimizer = make_optimizer(cfg, model)

    # *************** Dataset ****************
    train_dataset = build_dataset(config['spice']['data_train'])

    # TODO: batch size = 1?, num workers = 4?
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)

    dataset_val = build_dataset(config['spice']['data_test'])
    test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1)

    # TODO: Making sure the data set includes all needed transformations.

    train_ds, test_ds = [], []
    for data in train_loader:
        train_x, train_label = data[0], data[1]
        train_x = train_x[:, 0, :, :].unsqueeze(1)
        batch = {'input': train_x, 'output': train_label}
        train_ds.append(deepcopy(batch))

    for data in test_loader:
        train_x, train_label = data[0], data[1]
        train_x = train_x[:, 0, :, :].unsqueeze(1)
        batch = {'input': train_x, 'output': train_label}
        test_ds.append(deepcopy(batch))

    return train_ds, test_ds, model


def spice_model_build(config: ConfigWrapper, args):
    if args.weight:
        config['spice_model']['pretrained'] = args.weight

    # create model
    model = Sim2Sem(**config['spice_model'])
    print(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # model.eval()  # model.train(False)


    train_dataset = build_dataset(config[data_train])

    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=cfg.batch_size, shuffle=False, num_workers=1)

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

    return train_ds, test_ds, model


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
    elif args.datatype == 'tinyimagenet':
        train_ds, test_ds, model = build_model_for_cifar10(config, args, device)
    else:
        "implement on your own"
        pass
    return train_ds, test_ds, model
