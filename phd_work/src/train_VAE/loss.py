# Функция потерь
from torch import nn
import torch 
mse = torch.nn.MSELoss(reduction='mean')
def calc_det(x, mask_v = -10.0, start_stop_teken:bool = True, use_mask: bool = True):
    mask = torch.where(x != mask_v, 1, 0)
    if start_stop_teken:
        mask[:,0] = 0
        num_det = torch.sum(mask, dim=1)
        batch = num_det.shape[0]
        mask[torch.arange(batch),num_det[:,0],:] = 0
        num_det = torch.sum(mask, dim=1) 
        return mask
    if use_mask:
        # return all mask
        return mask
    else:
        # num_det - trnsor(num det in each event)
        num_det = torch.sum(mask, dim=1) # sum by det
        return num_det
def Num_Det_Loss(lengths_real: torch.Tensor, lenght_fake: torch.Tensor):
    # assert lengths_real.shape = 
    loss = mse(lengths_real, lenght_fake)
    return loss

def vae_loss(recon_x, x, mu, log_var, pred_num, mask = -10.0, use_mask: bool = True):
    recon_loss = nn.MSELoss(reduction='none')(recon_x, x) # return shape - batch, det, featch
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # подсчет кол-ва детекторов в событии
    # mask = calc_det(x, mask)
    if use_mask:
        num_det_mask = calc_det(x, mask, use_mask=use_mask)
        recon_loss*=num_det_mask
        num_det = torch.sum(num_det_mask, dim=1)[:,0][:,None, None]
        recon_loss = torch.sum(recon_loss/num_det, dim=1) # mean by active det
        recon_loss = torch.mean(recon_loss) # mean by batch and featches
        # loss for predict num active detections
        num_det_loss = Num_Det_Loss(num_det[:,:,0].float(), pred_num)
        return recon_loss, kl_divergence / x.size(0), num_det_loss
    else:
        recon_loss = torch.mean(recon_loss) # mean by active det
        return recon_loss, kl_divergence / x.size(0), num_det_loss
