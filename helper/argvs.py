import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, ImageNet
from torchvision.datasets import ImageFolder

from models.dualprompt import DualPrompt
from models.L2P import L2P
from models.EViT import EViT
from models.CertViT import CertViT
from models.PrEL2P import PrEL2P
from models.CertL2P import CertL2P
from models.ContrastiveL2P import ContrastiveL2P
from models.NotL2P import NotL2P
from models.GausskeyL2P import GausskeyL2P
from datas.CUB200 import CUB200
from datas.Dataset5 import Dataset5


############################################################################
#                                                                          #
#  Parsing functions                                                       #
#                                                                          #
############################################################################ 

def model_parser(model_name : str, args : list):
    models = {
        "dualprompt"     : (DualPrompt, dualprompt),
        "l2p"            : (L2P, l2p),
        "notl2p"         : (NotL2P, l2p),
        "evit"           : (EViT, evit),
        "certvit"        : (CertViT, certvit),
        "certl2p"        : (CertL2P, certl2p),
        "prel2p"         : (PrEL2P, prel2p),
        "contrastivel2p" : (ContrastiveL2P, l2p),
        "gausskeyl2p"    : (GausskeyL2P, l2p),
    }
    try:
        return models[model_name][0], vars(models[model_name][1].parse_known_args(args)[0])
    except KeyError:
        raise ValueError("unknown model name {}".format(model_name))

def criterion_parser(criterion_name : str):
    criterions = {
        "crossentropy" : nn.CrossEntropyLoss,
        "bce"          : nn.BCEWithLogitsLoss,
        "bce_logits"   : nn.BCELoss,
        "mse"          : nn.MSELoss,
        "l1"           : nn.L1Loss,
        "custom"       : "custom",
    }
    try:
        return criterions[criterion_name]
    except KeyError:
        raise ValueError("unknown criterion name {}".format(criterion_name))

def optimizer_parser(optimizer_name : str, args : list):
    optimizers = {
        "sgd"      : (torch.optim.SGD, sgd),
        "adam"     : (torch.optim.Adam, adam),
        "adadelta" : (torch.optim.AdamW, adamw),
        "rmsprop"  : (torch.optim.RMSprop, rmsprop),
    }
    try:
        return optimizers[optimizer_name], vars(optimizers[optimizer_name].parse_args(args))
    except KeyError:
        raise ValueError("unknown optimizer name {}".format(optimizer_name))

def scheduler_parser(scheduler_name : str, args : list):
    schedulers = {
        "step"        : (torch.optim.lr_scheduler.StepLR, step),
        "exponential" : (torch.optim.lr_scheduler.ExponentialLR, exponential),
        "cosine"      : (torch.optim.lr_scheduler.CosineAnnealingLR, cosine),
        "const"       : (torch.optim.lr_scheduler.LambdaLR, const),
    }
    try:
        return schedulers[scheduler_name], vars(schedulers[scheduler_name].parse_args(args))
    except KeyError:
        raise ValueError("unknown scheduler name {}".format(scheduler_name))

def dataset(args, _data : str) -> Dataset:
    data = {
        "cifar10"      : (CIFAR10,      10),
        "cifar100"     : (CIFAR100,     100),
        "mnist"        : (MNIST,        10),
        "fashionmnist" : (FashionMNIST, 10),
        "cub200"       : (CUB200,       200),
        "imagenet"     : (ImageNet,     1000),
        "dataset5"     : (Dataset5,     5),
    }
    try:
        args.model_args["class_num"] = data[_data][1]
        return data[_data][0]
    except KeyError:
        raise ValueError("unknown dataset name {}".format(_data))

def parse_args(args : list):
    parse    = parser.parse_known_args(args)
    parse, _ = parse

    parse.model, parse.model_args = model_parser(parse.model, args)
    if parse.criterion == "custom":
        print("Using custom criterion, please specify the loss_fn in the model")
    parse.criterion = criterion_parser(parse.criterion)
    parse.optimizer, parse.optimizer_args = optimizer_parser(parse.optimizer, args)
    parse.scheduler, parse.scheduler_args = scheduler_parser(parse.scheduler, args)
    parse.dataset = dataset(parse, parse.dataset)

    return parse

############################################################################
#                                                                          #
#  Main Parser for Program                                                 #
#                                                                          #
############################################################################  
parser = argparse.ArgumentParser(description = 'Train and Evaluate Model')

parser.add_argument("--model"         , type=str, help="model name to train or evaluate")
parser.add_argument("--criterion"     , type=str, help="loss function to use for training",           default="adam")
parser.add_argument("--optimizer"     , type=str, help="optimizer to use for training",               default="crossentropy")
parser.add_argument("--scheduler"     , type=str, help="learning Rate Scheduler to use for training", default="const")

parser.add_argument("--batch-size"    , type=int, help="batch size of data")
parser.add_argument("--step-size"     , type=int, help="number of batches for accumulate gradient", default=1)
parser.add_argument("--epochs"        , type=int, help="iteration of dataset for training")
parser.add_argument("--log-frequency" , type=int, help="number of print for a epoch", default=1)

parser.add_argument("--task-governor" , type=str, default=None, help="setting of continual learning for multiple task")
parser.add_argument("--num-tasks"     , type=int, default=1,help="task numbers")

parser.add_argument("--num-workers"   , type=int, default=2,help="task workers")
parser.add_argument("--dataset"       , type=str, help="number of print for a epoch")
parser.add_argument("--dataset-path"  , type=str, default="/home/datasets/", help="path of dataset")
parser.add_argument("--save-path"     , type=str, default="saved/model/", help="path to save model")

parser.add_argument("--eval"    , default=False, action= argparse.BooleanOptionalAction, help="perform reproduction not training if True")
parser.add_argument("--seed"    , type=int, default=None, help="manually set random seed")
parser.add_argument("--device"  , type=str, default='cuda', help="device to use for training/testing")
parser.add_argument("--pin-mem" , default=False, action= argparse.BooleanOptionalAction, help="use pin memory for data loader")
parser.add_argument("--use-amp" , default=False, action= argparse.BooleanOptionalAction, help="use amp for fp16")
parser.add_argument("--use-tf"  , default=False, action= argparse.BooleanOptionalAction, help="use tensorboard")
parser.add_argument("--debug"   , default=False, action= argparse.BooleanOptionalAction, help="in debug mode, program will shows more information")

parser.add_argument('--world-size',   default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank',         default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url',     default='env://', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--local-rank',   default=0, type=int, help='local rank for distributed training')

############################################################################
#                                                                          #
#  Model Parser for Each Model                                             #
#                                                                          #
############################################################################  

# L2P Parser
l2p = argparse.ArgumentParser(add_help=False)
l2p.add_argument("--backbone-name",  type=str)
l2p.add_argument("--pool-size",      type=int,   default=10)
l2p.add_argument("--selection-size", type=int,   default=5)
l2p.add_argument("--prompt-len",     type=int,   default=5)
l2p.add_argument("--lambda",         type=float, default=0.5)
l2p.add_argument("--xi",             type=float, default=0.1)
l2p.add_argument("--tau",            type=float, default=0.5)
l2p.add_argument("--zetta",          type=float, default=0.1)
l2p.add_argument("--_batchwise_selection" , default=True,  action= argparse.BooleanOptionalAction)
l2p.add_argument("--_diversed_selection"  , default=True,  action= argparse.BooleanOptionalAction)
l2p.add_argument("--_unsim_penalty"       , default=True,  action= argparse.BooleanOptionalAction)
l2p.add_argument("--_scale_prompts"       , default=True,  action= argparse.BooleanOptionalAction)
l2p.add_argument("--_scale_simmilarity"   , default=True,  action= argparse.BooleanOptionalAction)
l2p.add_argument("--_update_per_iter"     , default=False, action= argparse.BooleanOptionalAction)

# DualPrompt Parser
dualprompt = argparse.ArgumentParser(add_help=False)
dualprompt.add_argument("--backbone-name", type=str)
dualprompt.add_argument("--pos-g-prompt" , type=int, default = [1], nargs='+')
dualprompt.add_argument("--len-g-prompt" , type=int)
dualprompt.add_argument("--pos-e-prompt" , type=int, default = [2], nargs='+')
dualprompt.add_argument("--len-e-prompt" , type=int)
dualprompt.add_argument("--lambda"       , type=float, default=1.0)
dualprompt.add_argument("--prompt-func"  , type=str)

# CertL2P Parser
certl2p = argparse.ArgumentParser(parents=(l2p,), add_help=False)
certl2p.add_argument("--selection-layer", type=int,   default = [3,6,9], nargs='+')
certl2p.add_argument("--reserve-rate",    type=float, default = 0.7)

# EL2P Parser
prel2p = argparse.ArgumentParser(parents=(l2p,), add_help=False)
prel2p.add_argument("--selection-layer",  type=int,   default = [3,6,9], nargs='+')
prel2p.add_argument("--reserve-rate",     type=float, default = 0.7)

# EViT Parser
evit = argparse.ArgumentParser()
evit.add_argument("--backbone-name",      type=str)
evit.add_argument("--selection-layer",    type=int, default = [3,6,9], nargs='+')
evit.add_argument("--reserve-rate",       type=float, default = 0.7)

# CertViT Parser
certvit = argparse.ArgumentParser()
certvit.add_argument("--backbone-name",   type=str)
certvit.add_argument("--selection-layer", type=int, default = [3,6,9], nargs='+')
certvit.add_argument("--reserve-rate",    type=float, default = 0.7)

############################################################################
#                                                                          #
#  Optimizer Parser for Each                                               #
#                                                                          #
############################################################################ 

# Adam Parser
adam = argparse.ArgumentParser()
adam.add_argument("--lr"           , type=float, default=0.001)
adam.add_argument("--betas"        , type=tuple, default=(0.9, 0.999))
adam.add_argument("--eps"          , type=float, default=1e-08)
adam.add_argument("--weight_decay" , type=float, default=0)
adam.add_argument("--amsgrad"      , type=bool, default=False)

# Adamw Parser
adamw = argparse.ArgumentParser()
adamw.add_argument("--lr"           , type=float, default=0.001)
adamw.add_argument("--betas"        , type=tuple, default=(0.9, 0.999))
adamw.add_argument("--eps"          , type=float, default=1e-08)
adamw.add_argument("--weight_decay" , type=float, default=0)
adamw.add_argument("--amsgrad"      , type=bool, default=False)

# SGD Parser
sgd = argparse.ArgumentParser()
sgd.add_argument("--lr"           , type=float, default=0.001)
sgd.add_argument("--momentum"     , type=float, default=0.9)
sgd.add_argument("--dampening"    , type=float, default=0)
sgd.add_argument("--weight_decay" , type=float, default=0)
sgd.add_argument("--nesterov"     , type=bool, default=False)

# RMSProp Parser
rmsprop = argparse.ArgumentParser()
rmsprop.add_argument("--lr"           , type=float, default=0.001)
rmsprop.add_argument("--alpha"        , type=float, default=0.99)
rmsprop.add_argument("--eps"          , type=float, default=1e-08)
rmsprop.add_argument("--weight_decay" , type=float, default=0)
rmsprop.add_argument("--momentum"     , type=float, default=0)
rmsprop.add_argument("--centered"     , type=bool, default=False)

############################################################################
#                                                                          #
#  Scheduler Parser for Each                                               #
#                                                                          #
############################################################################ 

# ConstantLR Parser
const = argparse.ArgumentParser()
const.add_argument("--factor"       , type=float, default=1)
const.add_argument("--total-iters"  , type=int, default=100)
const.add_argument("--last-epoch"   , type=int, default=-1)

# ExponentialLR Parser
exponential = argparse.ArgumentParser()
exponential.add_argument("--gamma"        , type=float, default=0.1)
exponential.add_argument("--last-epoch"   , type=int, default=0)

# CosineAnnealingLR Parser
cosine = argparse.ArgumentParser()
cosine.add_argument("--T-max"        , type=int,   default=15)
cosine.add_argument("--eta-min"      , type=float, default=1e-6)
cosine.add_argument("--last-epoch"  , type=int,   default=-1)

# StepLR Parser
step = argparse.ArgumentParser()
step.add_argument("--step-size"      , type=int, default=1)
step.add_argument("--gamma"          , type=float, default=0.1)
step.add_argument("--last-epoch"     , type=int, default=0)
