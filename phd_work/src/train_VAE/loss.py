# Функция потерь
from torch import nn
import torch 

def vae_loss(recon_x, x, mu, log_var):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss, kl_divergence / x.size(0)