import sys
from tkinter.tix import Tree
import torch
import torch.nn as nn
from models.L2P import L2P
from models.dualprompt import DualPrompt
import argparse

#Functions to parse arguments

def model_parser(model_name : str, args : list):
    if model_name == "dualprompt":
        return DualPrompt, dualprompt.parse_known_args(args)
    elif model_name == "l2p":
        return L2P, l2p.parse_known_args(args)
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
        return torch.optim.Adam, adam.parse_known_args(args)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW, adamw.parse_known_args(args)
    elif optimizer_name == "sgd":
        return torch.optim.SGD, sgd.parse_known_args(args)
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop, rmsprop.parse_known_args(args)
    else:
        raise ValueError("unknown optimizer name {}".format(optimizer_name)[0])

def scheduler_parser(scheduler_name : str, args : list):
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR, step.parse_known_args(args)
    elif scheduler_name == "const":
        return torch.optim.lr_scheduler.ConstantLR, const.parse_known_args(args)
    elif scheduler_name == "exp":
        return torch.optim.lr_scheduler.ExponentialLR, exp.parse_known_args(args)
    elif scheduler_name == "cos":
        return torch.optim.lr_scheduler.CosineAnnealingLR, cos.parse_known_args(args)
    else:
        raise ValueError("unknown scheduler name {}".format(scheduler_name))

def parse_args(args : list):
    parse = parser.parse_known_args(args)
    parse, args = parse
    parse = vars(parse)
    
    parse["model"], (parse["model_args"], args) = model_parser(parse["model"], args)
    parse["model_args"] = vars(parse["model_args"])

    if parse["criterion"] == "custom": pass
    else: parse["criterion"] = criterion_parser(parse["criterion"])

    parse["optimizer"], (parse["optimizer_args"], args) = optimizer_parser(parse["optimizer"], args)
    parse["optimizer_args"] = vars(parse["optimizer_args"])

    parse["scheduler"], (parse["scheduler_args"], args) = scheduler_parser(parse["scheduler"], args)
    parse["scheduler_args"] = vars(parse["scheduler_args"])

    return parse

def parse_log_settings(args : list):
    parse = log_parser.parse_known_args(args)
    return parse


############################################################################
#                                                                          #
#  Main Parser for Program                                                 #
#                                                                          #
############################################################################  
parser = argparse.ArgumentParser(description = 'Train and Evaluate Model')

parser.add_argument("-model"          , type=str, help="model to train")
parser.add_argument("-criterion"      , type=str, help="loss function to use for training")
parser.add_argument("-optimizer"      , type=str, help="optimizer to use for training")
parser.add_argument("-scheduler"      , type=str, help="learning Rate Scheduler to use for training")

parser.add_argument("-batchsize"      , type=int, help="batch size of data")
parser.add_argument("-stepsize"       , type=int, help="number of batches for accumulate gradient")
parser.add_argument("-epochs"         , type=int, help="iteration of dataset for training")
parser.add_argument("-log_freqency"   , type=int, help="number of print for a epoch")

parser.add_argument("-num_tasks"      , type=int, default=1,help="task numbers")
parser.add_argument("-task_governor"  , type=str, default=None, help="setting of continual learning for multiple task")

parser.add_argument("-dataset"        , type=str, help="number of print for a epoch")
parser.add_argument("-worldsize"      , type=int, default=-1, help="world size of distributed training")
parser.add_argument("-num_workers"    , type=int, default=2, help="number of workers for data loader")
parser.add_argument("--dataset_path" , type=str, default="/home/datasets/", help="iteration of dataset for training")
parser.add_argument("--save_path"    , type=str, default="saved/model/", help="iteration of dataset for training")

parser.add_argument("-rank"          , type=int, default=-1, help="number of Rank for distributed training")
parser.add_argument("-dist_url"      , type=str, default="env://", help="distributed training url")
parser.add_argument("-dist_backend"  , type=str, default="gloo", help="distributed training backend")
parser.add_argument("--multiprocessing_distributed", default=False, action="store_const", const = True, help="use or not multiprocessing distributed")

parser.add_argument("-dataset_path" , type=str, default="/home/datasets/", help="iteration of dataset for training")
parser.add_argument("-save_path"    , type=str, default="saved/model/", help="iteration of dataset for training")

parser.add_argument("-seed"     , type=int, default=None, help="manually set random seed")
parser.add_argument("--use_amp" , default=False, action="store_const", const = True, help="use amp for fp16")
parser.add_argument("--debug"   , default=False, action="store_const", const = True, help="in debug mode, program will shows more information")

log_parser = argparse.ArgumentParser(description = 'Logging Options')
log_parser.add_argument("--file"    , type=str, default=None, nargs='+', help="log print into file with name")
log_parser.add_argument("--console" , default=False, action="store_const", const = True, help="log print into console")
log_parser.add_argument("--str"     , default=False, action="store_const", const = True, help="log print into string")

############################################################################
#  Model Parser for Each Model                                             #
############################################################################  

# DualPrompt Parser
dualprompt = argparse.ArgumentParser()
dualprompt.add_argument("-pos_g_prompt" , type=int, default = 1, nargs='+')
dualprompt.add_argument("-len_g_prompt" , type=int)
dualprompt.add_argument("-pos_e_prompt" , type=int, default = 2, nargs='+')
dualprompt.add_argument("-len_e_prompt" , type=int)
dualprompt.add_argument("-prompt_func"  , type=str)
dualprompt.add_argument("-backbone_name", type=str)

# L2P Parser
l2p = argparse.ArgumentParser()
l2p.add_argument("-backbone_name" , type=str)
l2p.add_argument("-pool_size"     , type=int, default=10)
l2p.add_argument("-selection_size", type=int, default=5)
l2p.add_argument("-prompt_len"    , type=int, default=5)

############################################################################
#  Optimizer Parser for Each                                               #
############################################################################ 
 
# Adam Parser
adam = argparse.ArgumentParser()
adam.add_argument("-lr"           , type=float, default=0.001)
adam.add_argument("-betas"        , type=tuple, default=(0.9, 0.999))
adam.add_argument("-eps"          , type=float, default=1e-08)
adam.add_argument("-weight_decay" , type=float, default=0)
adam.add_argument("-amsgrad"      , type=bool, default=False)

# Adamw Parser
adamw = argparse.ArgumentParser()
adamw.add_argument("-lr"           , type=float, default=0.001)
adamw.add_argument("-betas"        , type=tuple, default=(0.9, 0.999))
adamw.add_argument("-eps"          , type=float, default=1e-08)
adamw.add_argument("-weight_decay" , type=float, default=0)
adamw.add_argument("-amsgrad"      , type=bool, default=False)

# SGD Parser
sgd = argparse.ArgumentParser()
sgd.add_argument("-lr"           , type=float, default=0.001)
sgd.add_argument("-momentum"     , type=float, default=0.9)
sgd.add_argument("-weight_decay" , type=float, default=0)
sgd.add_argument("-nesterov"     , type=bool, default=False)

# RMSProp Parser
rmsprop = argparse.ArgumentParser()
rmsprop.add_argument("-lr"           , type=float, default=0.001)
rmsprop.add_argument("-alpha"        , type=float, default=0.99)
rmsprop.add_argument("-eps"          , type=float, default=1e-08)
rmsprop.add_argument("-weight_decay" , type=float, default=0)
rmsprop.add_argument("-centered"     , type=bool, default=False)

############################################################################
#  Scheduler Parser for Each                                               #
############################################################################ 

# ConstantLR Parser
const = argparse.ArgumentParser()
const.add_argument("-factor"       , type=float, default=1)
const.add_argument("-total_iters"  , type=int, default=100)
const.add_argument("-last_epoch"   , type=int, default=-1)

# ExponentialLR Parser
exp = argparse.ArgumentParser()
exp.add_argument("-lr"           , type=float, default=0.001)
exp.add_argument("-step_size"    , type=int, default=1)
exp.add_argument("-gamma"        , type=float, default=0.1)
exp.add_argument("-power"        , type=float, default=0.9)
exp.add_argument("-last_epoch"   , type=int, default=0)

# CosineAnnealingLR Parser
cos = argparse.ArgumentParser()
cos.add_argument("-lr"           , type=float, default=0.001)
cos.add_argument("-step_size"    , type=int, default=1)
cos.add_argument("-gamma"        , type=float, default=0.1)
cos.add_argument("-warmup_steps" , type=int, default=0)
cos.add_argument("-warmup_init_lr", type=float, default=0.001)
cos.add_argument("-warmup_gamma" , type=float, default=0.1)
cos.add_argument("-target_lr"    , type=float, default=0.001)
cos.add_argument("-last_epoch"   , type=int, default=0)

# StepLR Parser
step = argparse.ArgumentParser()
step.add_argument("-lr"             , type=float, default=0.001)
step.add_argument("-step_size"      , type=int, default=1)
step.add_argument("-gamma"          , type=float, default=0.1)
step.add_argument("-warmup_steps"   , type=int, default=0)
step.add_argument("-warmup_init_lr" , type=float, default=0.001)
step.add_argument("-warmup_gamma"   , type=float, default=0.1)
step.add_argument("-target_lr"      , type=float, default=0.001)
step.add_argument("-last_epoch"     , type=int, default=0)