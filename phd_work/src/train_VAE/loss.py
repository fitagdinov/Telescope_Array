# Функция потерь
from torch import nn
from typing import Optional
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
def CE_loss_particle(pred : torch.Tensor, real : torch.Tensor):
    # assert pred.shape = real.shape
    loss = nn.CrossEntropyLoss(reduction='none')(pred, real)
    return torch.mean(loss)
def vae_loss(recon_x, x, mu, log_var, pred_num, pred_part, real_part, mask = -10.0, use_mask: bool = True, koef_loss: Optional[torch.Tensor] = None,
             reduce_loss_per_event :bool = False):
    if koef_loss is None:
        koef_loss = torch.ones(1,6)
    koef_loss = koef_loss.unsqueeze(0)
    recon_loss = nn.MSELoss(reduction='none')(recon_x, x) # return shape - batch, det, featch
    recon_loss *= koef_loss
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # подсчет кол-ва детекторов в событии
    # mask = calc_det(x, mask)
    if use_mask:
        num_det_mask = calc_det(x, mask, use_mask=use_mask)
        recon_loss*=num_det_mask
        num_det = torch.sum(num_det_mask, dim=1)[:,0][:,None, None]
        recon_loss = torch.sum(recon_loss/num_det, dim=1) # mean by active det
        # print(recon_loss.shape, 'recon_loss')
        if not(reduce_loss_per_event):
            recon_loss = torch.mean(recon_loss) # mean by batch and featches
        # loss for predict num active detections
        num_det_loss = Num_Det_Loss(num_det[:,:,0].float(), pred_num)

        # loss for predict particles
        loss_mass = CE_loss_particle(pred_part, real_part)
        return recon_loss, kl_divergence / x.size(0), num_det_loss, loss_mass
    else:
        recon_loss = torch.mean(recon_loss) # mean by active det
        return recon_loss, kl_divergence / x.size(0), num_det_loss, loss_mass
