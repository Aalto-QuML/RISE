import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
from model_utils import *
from torch.distributions import MultivariateNormal
from helpers import *
import torch.distributions as dists

class CNN_LNP(nn.Module): 
    
    def __init__(self,num_channels):
        super().__init__()
        self.channel = num_channels
        self.hidden = 64
        self.conv_theta = nn.Conv2d(self.channel, self.hidden, 9, 1, 4)

        self.cnn = nn.Sequential(
            nn.Conv2d(self.hidden//4 + 2*self.hidden, self.hidden//2, 1, 1, 0),
            Conv2dResBlock(self.hidden//2, self.hidden//2),
            Conv2dResBlock(self.hidden//2, self.hidden//2),
            Conv2dResBlock(self.hidden//2, self.hidden//2),
            Conv2dResBlock(self.hidden//2, self.hidden//2),
            Conv2dResBlock(self.hidden//2, self.hidden//2),
            nn.Conv2d(self.hidden//2, 2*self.channel, 1, 1, 0)
        )

        self.pos = nn.Softplus()

        self.latent_encoder = nn.Sequential(nn.Conv2d(2*self.hidden, self.hidden//2, 3, 1, 1),nn.ReLU(),nn.Conv2d(self.hidden//2, self.hidden//2, 1, 1, 0),)
        self.q_z_loc_transformer = nn.Identity()
        self.q_z_scale_transformer=lambda z_scale: 0.1 + 0.9 * torch.sigmoid(z_scale)
        self.LatentDistribution = MultivariateNormalDiag
        self.n_z_samples = 1


    def infer_latents(self,R):

        q_z_suffstat = self.latent_encoder(R)
        q_z_loc, q_z_scale = q_z_suffstat[:,:self.hidden//4],q_z_suffstat[:,self.hidden//4:]

        q_z_loc = self.q_z_loc_transformer(q_z_loc)
        q_z_scale = self.q_z_scale_transformer(q_z_scale)

        # batch shape = [batch_size, *n_lat] ; event shape = [z_dim]
        q_zCc = self.LatentDistribution(q_z_loc, q_z_scale)

        return q_zCc



    def forward(self,x_trgt,mask):
        #### Context to induced
        B,_,N,_ = x_trgt.shape
        #x_trgt = x_trgt.unsqueeze(dim=1)
        #mask = mask.unsqueeze(dim=1)
        signal = x_trgt*mask
        density = mask

        signal = self.conv_theta(signal)
        density = self.conv_theta(density)

        R = torch.cat([signal, density], 1)

        q_z = self.infer_latents(R)
        z_samples = q_z.rsample([self.n_z_samples])
        final = torch.cat([R,z_samples.squeeze(dim=0)],dim=1)
        f = self.cnn(final)
        mean, std = f.split(self.channel, 1)
        std = self.pos(std)

        dist = dists.MultivariateNormal(loc=mean, scale_tril= torch.diag_embed(std))

        return mean,std,dist
        




class CNN_LNP_MNAR(nn.Module): 
    
    def __init__(self,num_channels):
        super().__init__()
        self.channel = num_channels
        self.hidden = 64
        self.conv_theta = nn.Conv2d(self.channel, self.hidden, 9, 1, 4)

        self.cnn = nn.Sequential(
            nn.Conv2d(self.hidden//4 + 2*self.hidden, self.hidden//2, 1, 1, 0),
            Conv2dResBlock(self.hidden//2, self.hidden//2),
            Conv2dResBlock(self.hidden//2, self.hidden//2),
            Conv2dResBlock(self.hidden//2, self.hidden//2),
            Conv2dResBlock(self.hidden//2, self.hidden//2),
            Conv2dResBlock(self.hidden//2, self.hidden//2),
            nn.Conv2d(self.hidden//2, 2*self.channel, 1, 1, 0)
        )

        self.pos = nn.Softplus()

        self.mask_ = nn.Sequential(
            nn.Conv2d(2*self.hidden, self.hidden//2, 1, 1, 0),
            Conv2dResBlock(self.hidden//2, self.hidden//2),
            Conv2dResBlock(self.hidden//2, self.hidden//2),
            Conv2dResBlock(self.hidden//2, self.hidden//2),
            nn.Conv2d(self.hidden//2, self.channel, 1, 1, 0),
            nn.Sigmoid()
        )

        self.latent_encoder = nn.Sequential(nn.Conv2d(2*self.hidden, self.hidden//2, 3, 1, 1),nn.ReLU(),nn.Conv2d(self.hidden//2, self.hidden//2, 1, 1, 0),)
        self.q_z_loc_transformer = nn.Identity()
        self.q_z_scale_transformer=lambda z_scale: 0.1 + 0.9 * torch.sigmoid(z_scale)
        self.LatentDistribution = MultivariateNormalDiag
        self.n_z_samples = 1


    def infer_latents(self,R):

        q_z_suffstat = self.latent_encoder(R)
        q_z_loc, q_z_scale = q_z_suffstat[:,:self.hidden//4],q_z_suffstat[:,self.hidden//4:]

        q_z_loc = self.q_z_loc_transformer(q_z_loc)
        q_z_scale = self.q_z_scale_transformer(q_z_scale)

        # batch shape = [batch_size, *n_lat] ; event shape = [z_dim]
        q_zCc = self.LatentDistribution(q_z_loc, q_z_scale)

        return q_zCc



    def forward(self,x_trgt,mask):
        #### Context to induced
        #breakpoint()
        B,_,N,_ = x_trgt.shape
        #x_trgt = x_trgt.unsqueeze(dim=1)
        #mask = mask.unsqueeze(dim=1)
       
        signal = x_trgt*mask
        density = mask

        signal = self.conv_theta(signal.float())
        density = self.conv_theta(density.float())
        
        R = torch.cat([signal, density], 1)
        

        mask_pred = self.mask_(R)

        q_z = self.infer_latents(R)
        z_samples = q_z.rsample([self.n_z_samples])
        final = torch.cat([R,z_samples.squeeze(dim=0)],dim=1)
        f = self.cnn(final)
        mean, std = f.split(self.channel, 1)
        std = self.pos(std)

        dist = dists.MultivariateNormal(loc=mean, scale_tril= torch.diag_embed(std))

        return mean,std,dist, mask_pred
        


class MLP_LNP(nn.Module): 
    
    def __init__(self,num_channels):
        super().__init__()
        self.channel = num_channels
        self.hidden = 64
        self.act = nn.ReLU()
        self.conv_theta = nn.Sequential(nn.Linear(self.channel, self.hidden//2),self.act,nn.Linear(self.hidden//2, self.hidden))

        self.cnn = nn.Sequential(
            nn.Linear(self.hidden//4 + 2*self.hidden, self.hidden//2),self.act,
            nn.Linear(self.hidden//2, self.hidden//2),self.act,
            nn.Linear(self.hidden//2, self.hidden//2),self.act,
            nn.Linear(self.hidden//2, self.hidden//2),self.act,
            nn.Linear(self.hidden//2, self.hidden//2),self.act,
            nn.Linear(self.hidden//2, self.hidden//2),self.act,
            nn.Linear(self.hidden//2, 2*self.channel)
        )

        self.pos = nn.Softplus()

        self.latent_encoder = nn.Sequential(nn.Linear(2*self.hidden, self.hidden//2),self.act,nn.Linear(self.hidden//2, self.hidden//2),)
        self.q_z_loc_transformer = nn.Identity()
        self.q_z_scale_transformer=lambda z_scale: 0.1 + 0.9 * torch.sigmoid(z_scale)
        self.LatentDistribution = MultivariateNormalDiag
        self.n_z_samples = 1


    def infer_latents(self,R):
        #breakpoint()
        q_z_suffstat = self.latent_encoder(R)
        q_z_loc, q_z_scale = q_z_suffstat[:,:self.hidden//4],q_z_suffstat[:,self.hidden//4:]

        q_z_loc = self.q_z_loc_transformer(q_z_loc)
        q_z_scale = self.q_z_scale_transformer(q_z_scale)

        # batch shape = [batch_size, *n_lat] ; event shape = [z_dim]
        q_zCc = self.LatentDistribution(q_z_loc, abs(q_z_scale))

        return q_zCc



    def forward(self,x_trgt,mask):
        #### Context to induced
        B,N = x_trgt.shape
        #x_trgt = x_trgt.unsqueeze(dim=1)
        #mask = mask.unsqueeze(dim=1)
        signal = x_trgt*mask
        density = mask

        signal = self.conv_theta(signal)
        density = self.conv_theta(density)
        
        R = torch.cat([signal, density], -1)
        q_z = self.infer_latents(R)
        z_samples = q_z.rsample([self.n_z_samples])
        #print(R.shape,z_samples.shape)
        final = torch.cat([R,z_samples.squeeze(dim=0)],dim=-1)
        f = self.cnn(final)
        mean, std = f.split(self.channel, 1)
        std = self.pos(std)

        dist = dists.MultivariateNormal(loc=mean, scale_tril= torch.diag_embed(std))

        return mean,std,dist




class MLP_LNP_MNAR(nn.Module): 
    
    def __init__(self,num_channels):
        super().__init__()
        self.channel = num_channels
        self.hidden = 64
        self.act = nn.ReLU()
        self.conv_theta = nn.Sequential(nn.Linear(self.channel, self.hidden//2),self.act,nn.Linear(self.hidden//2, self.hidden))

        self.cnn = nn.Sequential(
            nn.Linear(self.hidden//4 + 2*self.hidden, self.hidden//2),self.act,
            nn.Linear(self.hidden//2, self.hidden//2),self.act,
            nn.Linear(self.hidden//2, self.hidden//2),self.act,
            nn.Linear(self.hidden//2, self.hidden//2),self.act,
            nn.Linear(self.hidden//2, self.hidden//2),self.act,
            nn.Linear(self.hidden//2, self.hidden//2),self.act,
            nn.Linear(self.hidden//2, 2*self.channel)
        )

        self.mask_ = nn.Sequential(
            nn.Linear(2*self.hidden, self.hidden//2),self.act,
            nn.Linear(self.hidden//2, self.hidden//2),self.act,
            nn.Linear(self.hidden//2, self.hidden//2),self.act,
            nn.Linear(self.hidden//2, self.hidden//2),self.act,
            nn.Linear(self.hidden//2, self.hidden//2),self.act,
            nn.Linear(self.hidden//2, self.hidden//2),self.act,
            nn.Linear(self.hidden//2, self.channel),
            nn.Sigmoid()
        )

        self.pos = nn.Softplus()

        self.latent_encoder = nn.Sequential(nn.Linear(2*self.hidden, self.hidden//2),self.act,nn.Linear(self.hidden//2, self.hidden//2),)
        self.q_z_loc_transformer = nn.Identity()
        self.q_z_scale_transformer=lambda z_scale: 0.1 + 0.9 * torch.sigmoid(z_scale)
        self.LatentDistribution = MultivariateNormalDiag
        self.n_z_samples = 1


    def infer_latents(self,R):
        #breakpoint()
        q_z_suffstat = self.latent_encoder(R)
        q_z_loc, q_z_scale = q_z_suffstat[:,:self.hidden//4],q_z_suffstat[:,self.hidden//4:]

        q_z_loc = self.q_z_loc_transformer(q_z_loc)
        q_z_scale = self.q_z_scale_transformer(q_z_scale)

        # batch shape = [batch_size, *n_lat] ; event shape = [z_dim]
        q_zCc = self.LatentDistribution(q_z_loc, abs(q_z_scale))

        return q_zCc



    def forward(self,x_trgt,mask):
        #### Context to induced
        B,N = x_trgt.shape
        #x_trgt = x_trgt.unsqueeze(dim=1)
        #mask = mask.unsqueeze(dim=1)
        signal = x_trgt*mask
        density = mask

        signal = self.conv_theta(signal.float())
        density = self.conv_theta(density.float())
        
        R = torch.cat([signal, density], -1)

        mask_pred = self.mask_(R)
        q_z = self.infer_latents(R)
        z_samples = q_z.rsample([self.n_z_samples])
        #print(R.shape,z_samples.shape)
        #breakpoint()
        final = torch.cat([R,z_samples.squeeze(dim=0)],dim=-1)
        f = self.cnn(final)
        mean, std = f.split(self.channel, 1)
        std = self.pos(std)

        dist = dists.MultivariateNormal(loc=mean, scale_tril= torch.diag_embed(std))

        return mean,std,dist,mask_pred







