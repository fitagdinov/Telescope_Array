# Функция потерь
from torch import nn
from typing import Optional
import torch 
mse = torch.nn.MSELoss(reduction='mean')

class Loss(nn.Module):
    def __init__(self,):
        super().__init__()
    def calc_det(self, x, mask_v = -10.0, start_stop_teken:bool = True, use_mask: bool = True):
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
    def Num_Det_Loss(self, lengths_real: torch.Tensor, lenght_fake: torch.Tensor):
        # assert lengths_real.shape = 
        loss = mse(lengths_real, lenght_fake)
        return loss
    def CE_loss_particle(self, pred : torch.Tensor, real : torch.Tensor):
        # assert pred.shape = real.shape
        loss = nn.CrossEntropyLoss(reduction='none')(pred, real)
        return torch.mean(loss)
    def vae_loss(self, recon_x, x, mu, log_var, pred_num, pred_part, real_part, mask = -10.0, use_mask: bool = True, koef_loss: Optional[torch.Tensor] = None,
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
            num_det_mask = self.calc_det(x, mask, use_mask=use_mask)
            recon_loss*=num_det_mask
            num_det = torch.sum(num_det_mask, dim=1)[:,0][:,None, None]
            recon_loss = torch.sum(recon_loss/num_det, dim=1) # mean by active det
            # print(recon_loss.shape, 'recon_loss')
            if not(reduce_loss_per_event):
                recon_loss = torch.mean(recon_loss) # mean by batch and featches
            # loss for predict num active detections
            num_det_loss = self.Num_Det_Loss(num_det[:,:,0].float(), pred_num)

            # loss for predict particles
            loss_mass = self.CE_loss_particle(pred_part, real_part)
            return recon_loss, kl_divergence / x.size(0), num_det_loss, loss_mass
        else:
            recon_loss = torch.mean(recon_loss) # mean by active det
            return recon_loss, kl_divergence / x.size(0), num_det_loss, loss_mass
        
    # from repo https://github.com/mperezcarrasco/PyTorch-DAGMM/blob/master/forward_step.py
    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        eps = 1e-12
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.n_gmm):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1))*eps).to(self.device)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())
        
        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(self.device)

        E_z = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi.unsqueeze(0)*E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)
        if sample_mean==True:
            E_z = torch.mean(E_z)            
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """ 
        # K: number of Gaussian mixture components
        # N: Number of samples
        # D: Latent dimension
        # z = NxD
        # gamma = NxK

        #phi = D
        phi = torch.sum(gamma, dim=0)/gamma.size(0) 

        #mu = KxD
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mu /= torch.sum(gamma, dim=0).unsqueeze(-1)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        
        #cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov
        

class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s
