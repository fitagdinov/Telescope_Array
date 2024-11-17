# Функция потерь
from torch import nn
import torch 
def calc_det(x, mask = -10):
    index = torch.where(x != mask)
    # num_det - trnsor(num det in each event)
    num_det = torch.unique(index[0], return_counts = True)[1]
    return num_det

def vae_loss(recon_x, x, mu, log_var, mask = -10):
    recon_loss = nn.MSELoss(reduction='none')(recon_x, x) # return shape - batch, det, featch
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # подсчет кол-ва детекторов в событии
    num_det = calc_det(x, mask)[:,None, None]
    recon_loss = torch.sum(recon_loss/num_det, dim=1) # mean by active det
    recon_loss = torch.mean(recon_loss) # mean by batch and featches
    return recon_loss, kl_divergence / x.size(0)