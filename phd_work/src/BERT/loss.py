# CORRECT

from torch import nn
from typing import Optional
import torch 

class MaskLoss():
    def __init__(self, reduction = 'mean'):
        # super().__init__()
        self.LossMSE = nn.MSELoss(reduction='none')
        self.Wall = 0.0
        self.weight = torch.tensor([0.3186702476079202, 0.6813297523920798]).to('cuda') # from validation
    

    def __call__(self, recon_x, x, mask_tensor, part=None):
        none_loss = self.LossMSE(recon_x, x)
        if part is None:
            weight = torch.ones_like(none_loss)
        else:
            weight = torch.where(part==1, self.weight[0], self.weight[1]).unsqueeze(-1).unsqueeze(-1) # batch, 1, 1
            weight = weight.to(none_loss.dtype).to(none_loss.device)
        none_loss_mask = none_loss*mask_tensor + self.Wall*none_loss
        none_loss_mask = none_loss_mask#*weight
        mean_loss =  torch.sum(none_loss_mask)/(torch.sum(mask_tensor) + 1e-6)

        return mean_loss

if __name__ == "__main__":
    # проверить лосс руками 
    x = torch.ones(2,4,6)
    recon = torch.rand(2,4,6)
    mask = torch.zeros_like(x).bool()
    mask[0,1] = True
    mask[1,2] = True
    print(x)
    print(mask)
    print(recon)
    Loss = MaskLoss()
    loss = Loss(x, recon, mask)
    print(loss)
    