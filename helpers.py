import os
from inference.base import *
from simulators.ricker import ricker
from simulators.oup import oup
from utils.torchutils import *
import matplotlib.pyplot as plt
import random
import sys
import sbibm
import torch
import numpy as np
import pickle
from torch.distributions.independent import Independent
from torch.distributions import Normal
import pandas as pd
cwd = os.getcwd()

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def channel_last(x):
    return x.transpose(1, 2).transpose(2, 3)


def simulate_data(simulator, prior, num_simulations):
    theta, x = simulate_for_sbi(simulator, prior, num_simulations=num_simulations)
    return theta, x


def MultivariateNormalDiag(loc, scale_diag):
    """Multi variate Gaussian with a diagonal covariance function (on the last dimension)."""
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return Independent(Normal(loc, scale_diag), 1)


def evaluate_imputation_mse(pred,truth,mask,max_val,min_val):

    anti_mask = 1-mask
    diff = (pred*anti_mask - truth*anti_mask)**2

    pred_unnorm = pred*(max_val - min_val) + min_val
    truth_unnorm = truth*(max_val - min_val) + min_val

    diff_unnorm = ( pred_unnorm *anti_mask - truth_unnorm*anti_mask)**2

    return torch.sqrt(diff.mean()), torch.sqrt(diff_unnorm.mean())



def compute_mean_imputation_mse(mask,trgt,train_trgt,max_val,min_val):
    mean_vals = train_trgt.mean(dim=0).repeat(mask.shape[0],1,1)
    mean_imputed = mean_vals*(1-mask)
    anti_mask = 1-mask
    diff = (mean_imputed - trgt*anti_mask)**2


    mean_imputed = mean_imputed*(max_val - min_val) + min_val
    trgt = trgt*(max_val - min_val) + min_val

    diff_unnorm = (mean_imputed - trgt*anti_mask)**2

    return torch.sqrt(diff.mean()), torch.sqrt(diff_unnorm.mean())


def nll(mean,std,truth):
    normal_lkl = torch.distributions.normal.Normal(mean, 1e-3 + std)
    lkl = - normal_lkl.log_prob(truth)
    loss_val = lkl.mean()
    #loss_val = torch.mean(lkl,dim=(0,1,3,4))
    return loss_val


def nll_logsumexp(mean,std,truth):
    normal_lkl = torch.distributions.normal.Normal(mean, 1e-3 + std)
    #breakpoint()
    lkl = - normal_lkl.log_prob(truth.repeat(1,4,1,1))
    loss_val = torch.logsumexp(lkl.sum(dim=(-1,-2)), 1).mean()
    #loss_val = torch.mean(lkl,dim=(0,1,3,4))
    return loss_val


def compute_zero_imputation_mse(mask,trgt,train_trgt,max_val,min_val):
    #mean_vals = train_trgt.mean(dim=0).repeat(mask.shape[0],1,1)
    mean_imputed = trgt*mask
    diff = (mean_imputed - trgt)**2

    mean_imputed = mean_imputed*(max_val - min_val) + min_val
    trgt = trgt*(max_val - min_val) + min_val

    diff_unnorm = (mean_imputed - trgt)**2

    return torch.sqrt(diff.mean()), torch.sqrt(diff_unnorm.mean())
