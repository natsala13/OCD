from pickle import FALSE
import torch
import torch.functional as F
from math import floor as floor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from copy import deepcopy
import os
import numpy as np
from diffusion_ocd import DiffusionModel, Model_Scale
from utils_OCD import overfitting_batch_wrapper, noising, generalized_steps, ConfigWrapper
import torch.utils.tensorboard as tb
from train import train, vgg_encode
from ema import EMAHelper
import os
import argparse
import json
from data_loader import wrapper_dataset


from train import calc_score
from tools.eval_semi import calculate_acc, calculate_nmi, calculate_ari


parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--eval', type=int, default=0,
    help='Evaluate models, default <0>')
parser.add_argument(
    '-t', '--train', type=int, default=1,
    help='train diffusion and scale models, default <1>')
parser.add_argument(
    '-rt', '--resume_training', type=int, default=0,
    help='resume train diffusion and scale models, default <0>')
parser.add_argument(
    '-pd', '--diffusion_model_path', type=str, default='',
    help='checkpoint path for diffusion model , default <''>')
parser.add_argument(
    '-ps', '--scale_model_path', type=str, default='',
    help='checkpoint path for scale model , default <''>')
parser.add_argument(
    '-l', '--learning_rate', type=float, default=2e-4, help='learning rate, default <2e-4>')
parser.add_argument(
    '-pb', '--backbone_path', type=str, default='./base_models/checkpoint_tinynerf.pt',
    help='checkpoint path for backbone, default </base_models/checkpoint_tinynerf.pt>')
parser.add_argument(
    '-pc', '--config_path', type=str, default='./configs/train_tinynerf.json',
    help='config path, default </configs/train_tinynerf.json>')
parser.add_argument(
    '-ptb', '--tensorboard_path', type=str, default='./logs',
    help='config path, default <./logs>')
parser.add_argument(
    '-pdtr', '--data_train_path', type=str, default='./data/tiny_nerf_data.npz',
    help='training data path, default <''/data/tiny_nerf_data.npz''>')
parser.add_argument(
    '-pdts', '--data_test_path', type=str, default='/data',
    help="test data path, default <'/data'>")
parser.add_argument(
    '-dt', '--datatype', type=str, default='tinynerf',
    help='datatype - tinynerf or not, default <tinynerf>')
parser.add_argument(
    '-prc', '--precompute_all', type=int, default=1,
    help='precomputation of overfitting to save time, default <1>')

##########################################################################################################
####################################### Configuration  ###################################################
##########################################################################################################

args = parser.parse_args()
print(args)
with open(args.config_path) as f:
    config = ConfigWrapper(**json.load(f))

# config = AttrDict(config)
torch.manual_seed(123456789)

##########################################################################################################
####################################### Parameters & Initializations #####################################
##########################################################################################################

module_path = args.backbone_path  # path to desired pretrained model
tb_path = args.tensorboard_path  # path to tensorboard log
tb_logger = tb.SummaryWriter(log_dir=tb_path)
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
lr = args.learning_rate  # learning rate for the diffusion model & scale estimation model

diffusion_model = DiffusionModel(config=config).to(device)
loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.CrossEntropyLoss()

scale_model = Model_Scale(config=config).to(device)
if args.resume_training:
    diffusion_model.load_state_dict(torch.load(args.diffusion_model_path))
    scale_model.load_state_dict(torch.load(args.scale_model_path))

train_loader, test_loader, model = wrapper_dataset(config, args, device)


# model.load_state_dict(torch.load(module_path, map_location=torch.device(device)))  # Load the pre-trained model.
model = model.to(device)

if config.training.loss == 'mse':
    opt_error_loss = torch.nn.MSELoss()
elif config.training.loss == 'ce':
    opt_error_loss = torch.nn.CrossEntropyLoss()
elif config.training.loss == 'own':
    # Change according to desired objective
    pass
optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=lr)
optimizer_scale = torch.optim.Adam(scale_model.parameters(), lr=2 * lr)
ema_helper = EMAHelper(mu=0.9999)
ema_helper.register(diffusion_model)

################################################# Check if weight is OK ##########################

print('* Loading weight for further work')
weight_name = config.model.weight_name  # TODO: Change name.
dmodel_original_weight = deepcopy(model.get_parameter(weight_name + '.weight'))
mat_shape = dmodel_original_weight.shape

assert len(mat_shape) == 2, "Weight to overfit should be a matrix !"
padding = []
for s in mat_shape:
    if s == 128:
        padding.append([0, 0])
    elif s < 128:
        rem = 128 - s
        if (rem % 2) == 0:
            padding.append([rem // 2, rem // 2])
        else:
            padding.append([rem // 2 + 1, rem // 2])
    elif (s % 2) == 1:
        padding.append([2, 1])
    else:
        padding.append([0, 0])


#################################################################################################
########################################### Train Phase #########################################
#################################################################################################
print('* Train start')
if args.train:
    diffusion_model, scale_model = train(args=args, config=config, optimizer=optimizer, optimizer_scale=optimizer_scale,
                                         device=device, diffusion_model=diffusion_model, scale_model=scale_model,
                                         model=model, train_loader=train_loader, padding=padding, mat_shape=mat_shape,
                                         ema_helper=ema_helper, tb_logger=tb_logger, loss_fn=loss_fn,
                                         opt_error_loss=opt_error_loss)
else:
    diffusion_model.load_my_state_dict(torch.load(args.diffusion_model_path, map_location=device))
    scale_model.load_my_state_dict(torch.load(args.scale_model_path, map_location=device))

#################################################################################################
########################################### Test Phase ##########################################
#################################################################################################

print('*' * 100)
ldiff, lopt, lbaseline = 0, 0, 0
idx = 0
predicted_labels = []
gt_labels = []

for train_x, train_label, _ in test_loader:
    batch = {'input': train_x.to(device), 'output': train_label.to(device)}

    # Overfitting encapsulation #
    weight, hfirst, outin = overfitting_batch_wrapper(
        datatype=args.datatype,
        bmodel=model, weight_name=weight_name,
        bias_name=weight_name,
        batch=batch, loss_fn=opt_error_loss,
        n_iteration=config.overfitting.n_overfitting,
        lr=config.overfitting.lr_overfitting,
        verbose=False
    )
    diff_weight = weight - dmodel_original_weight
    if args.datatype == 'tinynerf':
        encoding_out = vgg_encode(outin)
    else:
        encoding_out = outin

    with torch.no_grad():
        std = scale_model(hfirst, encoding_out)

    ldiffusion, loptimal, lbase, wdiff, label_predicted = generalized_steps(
        named_parameter=weight_name, numstep=config.diffusion.diffusion_num_steps_eval,
        x=(diff_weight.unsqueeze(0), hfirst, encoding_out), model=diffusion_model,
        bmodel=model, batch=batch, loss_fn=opt_error_loss,
        std=std, padding=padding,
        mat_shape=mat_shape, isnerf=(args.datatype == 'tinynerf')
    )

    ldiff += ldiffusion
    lopt += loptimal
    lbaseline += lbase

    predicted_labels += [label_predicted]
    gt_labels += [batch['output'].cpu().numpy()]

    print(
        f"\rBaseline loss {lbaseline / (idx + 1)}, Overfitted loss {lopt / (idx + 1)}, Diffusion loss {ldiff / (idx + 1)}",
        end='')

    idx += 1
    del batch['input']
    torch.cuda.empty_cache()

predicted_labels = np.concatenate(predicted_labels, axis=0)
gt_labels = np.concatenate(gt_labels, axis=0)

acc = calculate_acc(predicted_labels, gt_labels)
nmi = calculate_nmi(predicted_labels, gt_labels)
ari = calculate_ari(predicted_labels, gt_labels)
print('*************************')
print(f'Acc - {acc}, NMI - {nmi}, ARI - {ari}')



