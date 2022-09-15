import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

from models.dualprompt import DualPrompt
from models.L2P import L2P
from models.EL2P import EL2P
from models.ScaledL2P import ScaledL2P
from models.PrEL2P import PrEL2P
from models.CertL2P import CertL2P

#Functions to parse arguments

def model_parser(model_name : str, args : list):
    if model_name == "dualprompt":
        return DualPrompt, vars(dualprompt.parse_known_args(args)[0])
    elif model_name == "l2p":
        return L2P, vars(l2p.parse_known_args(args)[0])
    elif model_name == "el2p":
        return EL2P, vars(el2p.parse_known_args(args)[0])
    elif model_name == "certl2p":
        return CertL2P, vars(certl2p.parse_known_args(args)[0])
    elif model_name == "scaledl2p":
        return ScaledL2P, vars(scaledl2p.parse_known_args(args)[0])
    elif model_name == "prel2p":
        return PrEL2P, vars(prel2p.parse_known_args(args)[0])
    else:
        raise ValueError("unknown model name {}".format(model_name)[0])

def criterion_parser(criterion_name : str):
    if criterion_name == "crossentropy":
        return nn.CrossEntropyLoss
    elif criterion_name == "mse":
        return nn.MSELoss
    elif criterion_name == "bce":
        return nn.BCELoss
    else:
        raise ValueError("unknown criterion name {}".format(criterion_name))

def optimizer_parser(optimizer_name : str, args : list):
    if optimizer_name == "adam":
        return torch.optim.Adam,  vars(adam.parse_known_args(args)[0])
    elif optimizer_name == "adamw":
        return torch.optim.AdamW,  vars(adamw.parse_known_args(args)[0])
    elif optimizer_name == "sgd":
        return torch.optim.SGD,  vars(sgd.parse_known_args(args)[0])
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop,  vars(rmsprop.parse_known_args(args)[0])
    else:
        raise ValueError("unknown optimizer name {}".format(optimizer_name)[0])

def scheduler_parser(scheduler_name : str, args : list):
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR,  vars(step.parse_known_args(args)[0])
    elif scheduler_name == "const":
        return torch.optim.lr_scheduler.ConstantLR,  vars(const.parse_known_args(args)[0])
    elif scheduler_name == "exp":
        return torch.optim.lr_scheduler.ExponentialLR,  vars(exp.parse_known_args(args)[0])
    elif scheduler_name == "cos":
        return torch.optim.lr_scheduler.CosineAnnealingLR,  vars(cos.parse_known_args(args)[0])
    else:
        raise ValueError("unknown scheduler name {}".format(scheduler_name))

def dataset(args, _data : str) -> Dataset:
        if _data == 'CIFAR100':
            args.model_args["class_num"] = 100
            return CIFAR100
        elif _data == 'CIFAR10':
            args.model_args["class_num"] = 10
            return CIFAR10
        elif _data == 'MNIST':
            args.model_args["class_num"] = 10
            return MNIST
        elif _data == 'FashionMNIST':
            args.model_args["class_num"] = 10
            return FashionMNIST
        else:
            raise ValueError('Dataset {} not supported'.format(_data))

def parse_args(args : list):
    parse    = parser.parse_known_args(args)
    parse, _ = parse
    parse.model, parse.model_args = model_parser(parse.model, args)

    if parse.criterion == "custom":
        print("Using custom criterion, please specify the loss_fn in the model")
    else: parse.criterion = criterion_parser(parse.criterion)

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

parser.add_argument("--model"        , type=str, help="model to train")
parser.add_argument("--criterion"    , type=str, help="loss function to use for training")
parser.add_argument("--optimizer"    , type=str, help="optimizer to use for training")
parser.add_argument("--scheduler"    , type=str, help="learning Rate Scheduler to use for training")

parser.add_argument("--batch-size"    , type=int, help="batch size of data")
parser.add_argument("--step-size"     , type=int, help="number of batches for accumulate gradient")
parser.add_argument("--epochs"        , type=int, help="iteration of dataset for training")
parser.add_argument("--log-frequency" , type=int, help="number of print for a epoch")

parser.add_argument("--num-tasks"     , type=int, default=1,help="task numbers")
parser.add_argument("--task-governor" , type=str, default=None, help="setting of continual learning for multiple task")

parser.add_argument("--num-workers"   , type=int, default=2,help="task workers")
parser.add_argument("--dataset"       , type=str, help="number of print for a epoch")
parser.add_argument("--dataset-path"  , type=str, default="/home/datasets/", help="path of dataset")
parser.add_argument("--save-path"     , type=str, default="saved/model/", help="path to save model")

parser.add_argument("--seed"    , type=int, default=None, help="manually set random seed")
parser.add_argument("--device"  , type=str, default='cuda', help="device to use for training/testing")
parser.add_argument("--pin-mem" , default=False, action= argparse.BooleanOptionalAction, help="use pin memory for data loader")
parser.add_argument("--use-amp" , default=False, action= argparse.BooleanOptionalAction, help="use amp for fp16")
parser.add_argument("--debug"   , default=False, action= argparse.BooleanOptionalAction, help="in debug mode, program will shows more information")

# DDP configs:
parser.add_argument('--world-size',   default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank',         default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url',     default='env://', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--local-rank',   default=0, type=int, help='local rank for distributed training')

############################################################################
#  Model Parser for Each Model                                             #
############################################################################  

# DualPrompt Parser
dualprompt = argparse.ArgumentParser(add_help=False)
dualprompt.add_argument("--backbone-name", type=str)
dualprompt.add_argument("--pos-g-prompt" , type=int, default = [1], nargs='+')
dualprompt.add_argument("--len-g-prompt" , type=int)
dualprompt.add_argument("--pos-e-prompt" , type=int, default = [2], nargs='+')
dualprompt.add_argument("--len-e-prompt" , type=int)
dualprompt.add_argument("--lambda"       , type=float, default=1.0)
dualprompt.add_argument("--prompt-func"  , type=str)

# L2P Parser
l2p = argparse.ArgumentParser(add_help=False)
l2p.add_argument("--backbone-name",        type=str)
l2p.add_argument("--pool-size",            type=int, default=10)
l2p.add_argument("--selection-size",       type=int, default=5)
l2p.add_argument("--prompt-len",           type=int, default=5)
l2p.add_argument("--lambda",               type=float, default=0.5)
l2p.add_argument("--_cls-at-front" ,       default=True,  action= argparse.BooleanOptionalAction, help="set class token at front of prompt")
l2p.add_argument("--_batchwise-selection", default=True,  action= argparse.BooleanOptionalAction, help="batchwise selection for")
l2p.add_argument("--_mixed-prompt-order",  default=False, action= argparse.BooleanOptionalAction, help="randomize the order of prompt")
l2p.add_argument("--_mixed-prompt-token",  default=False, action= argparse.BooleanOptionalAction, help="randomize the order of prompt")
l2p.add_argument("--_learnable-pos-emb",   default=False,  action= argparse.BooleanOptionalAction, help="randomize the order of prompt")

# ScaledL2P Parser
scaledl2p = argparse.ArgumentParser(parents=(l2p,), add_help=False)
scaledl2p.add_argument("--tau",                default=1.0, type=float)
scaledl2p.add_argument("--_scale_simmilarity", default=False, action= argparse.BooleanOptionalAction, help="randomize the order of prompt")

# EL2P Parser
el2p = argparse.ArgumentParser(parents=(l2p,), add_help=False)
el2p.add_argument("--selection-layer",      type=int, default = [3,6,9], nargs='+')
el2p.add_argument("--reserve-rate",         type=float, default = 0.7)

# EL2P Parser
prel2p = argparse.ArgumentParser(parents=(l2p,), add_help=False)
prel2p.add_argument("--selection-layer",      type=int, default = [3,6,9], nargs='+')
prel2p.add_argument("--reserve-rate",         type=float, default = 0.7)

# CertL2P Parser
certl2p = argparse.ArgumentParser(parents=(l2p,), add_help=False)
certl2p.add_argument("--selection-layer",      type=int, default = [3,6,9], nargs='+')
certl2p.add_argument("--reserve-rate",         type=float, default = 0.7)

############################################################################
#  Optimizer Parser for Each                                               #
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
sgd.add_argument("--weight_decay" , type=float, default=0)
sgd.add_argument("--nesterov"     , type=bool, default=False)

# RMSProp Parser
rmsprop = argparse.ArgumentParser()
rmsprop.add_argument("--lr"           , type=float, default=0.001)
rmsprop.add_argument("--alpha"        , type=float, default=0.99)
rmsprop.add_argument("--eps"          , type=float, default=1e-08)
rmsprop.add_argument("--weight_decay" , type=float, default=0)
rmsprop.add_argument("--centered"     , type=bool, default=False)

############################################################################
#  Scheduler Parser for Each                                               #
############################################################################ 

# ConstantLR Parser
const = argparse.ArgumentParser()
const.add_argument("--factor"       , type=float, default=1)
const.add_argument("--total-iters"  , type=int, default=100)
const.add_argument("--last-epoch"   , type=int, default=-1)

# ExponentialLR Parser
exp = argparse.ArgumentParser()
exp.add_argument("--lr"           , type=float, default=0.001)
exp.add_argument("--step-size"    , type=int, default=1)
exp.add_argument("--gamma"        , type=float, default=0.1)
exp.add_argument("--power"        , type=float, default=0.9)
exp.add_argument("--last-epoch"   , type=int, default=0)

# CosineAnnealingLR Parser
cos = argparse.ArgumentParser()
cos.add_argument("--lr"           , type=float, default=0.001)
cos.add_argument("--step-size"    , type=int, default=1)
cos.add_argument("--gamma"        , type=float, default=0.1)
cos.add_argument("--warmup-steps" , type=int, default=0)
cos.add_argument("--warmup-init-lr", type=float, default=0.001)
cos.add_argument("--warmup-gamma" , type=float, default=0.1)
cos.add_argument("--target-lr"    , type=float, default=0.001)
cos.add_argument("--last-epoch"   , type=int, default=0)

# StepLR Parser
step = argparse.ArgumentParser()
step.add_argument("--lr"             , type=float, default=0.001)
step.add_argument("--step-size"      , type=int, default=1)
step.add_argument("--gamma"          , type=float, default=0.1)
step.add_argument("--warmup-steps"   , type=int, default=0)
step.add_argument("--warmup-init-lr" , type=float, default=0.001)
step.add_argument("--warmup-gamma"   , type=float, default=0.1)
step.add_argument("--target-lr"      , type=float, default=0.001)
step.add_argument("--last-epoch"     , type=int, default=0)
