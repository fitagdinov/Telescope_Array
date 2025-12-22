# CORRECT

from torch import nn
from typing import Optional
import torch 
import torch.nn.functional as F
class MaskLoss():
    def __init__(self,
            mul_sig:bool = False, 
            device = 'cuda',
            mul_len = False,
            Wall = 0.0,
            stop_token=-11,
            padding_value = -10):
        # super().__init__()
        self.LossMSE = nn.MSELoss(reduction='none')
        self.Wall = Wall
        self.weight = torch.tensor([0.3186702476079202, 0.6813297523920798]).to(device) # from validation
        self.mul_sig = mul_sig
        self.mul_len = mul_len
        self.stop_token = stop_token
        self.padding_value = padding_value
    def get_mask(self, x):
        """
        где 1 - там не вспомогательный токен, где 0 -вспомогательный
        вывод  bs, seq. 6
        """
        stop_token = self.stop_token
        padding_value = self.padding_value
        stop_token = torch.tensor(self.stop_token, dtype=torch.long, device=x.device)
        padding_value = torch.tensor(self.padding_value, dtype=torch.long, device=x.device)
        mask = torch.ones_like(x, dtype=torch.long)  # Убедитесь, что это long (int64)
        mask = torch.where(x == stop_token, torch.tensor(0, dtype=torch.long, device=x.device), mask)
        mask = torch.where(x == padding_value, torch.tensor(0, dtype=torch.long, device=x.device), mask)
        mask[:,0,:] = 0
        # не учитываем координаты
        mask[:,:,:3] = 0

        # unused MASK. BE  carefull
        return mask.bool().to(x.device)

    def __call__(self, recon_x, x, mask_tensor, part=None,
                reduction = 'mean'
                ):
        none_loss = self.LossMSE(recon_x, x)
        if self.mul_sig:
            min_sig, _ = torch.min(x[:,:,3:4], dim = 1) # b,1
            min_sig = min_sig.unsqueeze(1)# b,1,1
            # sig >= 0 
            sig = (x[:,:,3:4] - min_sig + 1e-6)
            none_loss *= sig
        if part is None:
            weight = torch.ones_like(none_loss)
        else:
            weight = torch.where(part==1, self.weight[0], self.weight[1]).unsqueeze(-1).unsqueeze(-1) # batch, 1, 1
            weight = weight.to(none_loss.dtype).to(none_loss.device)
        none_loss_mask = none_loss*mask_tensor
        # none_loss_mask = none_loss_mask#*weight
        mask = self.get_mask(x)
        if reduction != 'none':
            if self.mul_sig:
                mean_loss = torch.sum(none_loss_mask)/(torch.sum(sig) + 1e-6)
            else:
                mean_loss = torch.sum(none_loss_mask)/(torch.sum(mask_tensor) + 1e-6)
            wall_loss = torch.sum(none_loss*mask)/(torch.sum(mask) + 1e-6)
            loss = mean_loss + self.Wall*wall_loss
        else:
            # Исправляем: используем правильный знаменатель для каждого элемента в батче
            if self.mul_sig:
                mean_loss = torch.sum(none_loss_mask, dim=[1,2])/(torch.sum(sig, dim=[1,2]) + 1e-6)
            else:
                mean_loss = torch.sum(none_loss_mask, dim=[1,2])/(torch.sum(mask_tensor, dim=[1,2]) + 1e-6)
            wall_loss = torch.sum(none_loss*mask, dim=[1,2])/(torch.sum(mask, dim=[1,2]) + 1e-6)
            loss = mean_loss + self.Wall*wall_loss
        return loss
class ClassificationLoss(nn.Module):
    def __init__(self, num_class: int = 2, class_weights: Optional[torch.Tensor] = None, 
                 label_smoothing: float = 0.0, weight_decay: float = 0.0):
        super().__init__()
        self.num_class = num_class
        self.label_smoothing = label_smoothing
        self.weight_decay = weight_decay
        
        # Используем веса классов для борьбы с дисбалансом
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(
                weight=class_weights, 
                reduction='none',
                label_smoothing=label_smoothing
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                reduction='none',
                label_smoothing=label_smoothing
            )
    
    def forward(self, pred: torch.Tensor, real: torch.Tensor, model_params: Optional[list] = None):
        '''
        real has shape (b,)
        pred has shape (b, num_class)
        '''
        # Основная функция потерь
        loss = self.loss_fn(pred, real)
        
        # Добавляем регуляризацию L2
        if self.weight_decay > 0 and model_params is not None:
            l2_reg = 0
            for param in model_params:
                l2_reg += torch.norm(param, 2)
            loss = loss + self.weight_decay * l2_reg
        
        return torch.mean(loss)
class EmbadingLoss(nn.Module):
    def __init__(self):
        super().__init__()        
    def forward(self, pred: torch.Tensor,
                    real: torch.Tensor
                    ):
        """
        Здача - соединить похожие эмбэдинги и разьединить отличные
        pred - эмбэдинг с маскированием (batch, latent_dim)
        real - эмбэдинг без маскирования (batch, latent_dim)
        """
        assert pred.dim()==2 and real.dim()==2, "expect (B, D)"
        assert pred.size(0) == real.size(0), "batch sizes must match"
        pred_norm = F.normalize(pred, dim=1)
        real_norm = F.normalize(real, dim=1)
        close_matrix = pred_norm@real_norm.t() # batch, batch
        # close_matrix - матрица косинусного расстояния.
        # Идеал - единичная матрица
        target = torch.arange(close_matrix.size(0), device = pred_norm.device) 
        loss = F.cross_entropy(close_matrix, target)
        return loss


        

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
    Loss = MaskLoss(mul_sig = True, device='cpu')
    loss = Loss(x, recon, mask, reduction='none')
    print(loss, loss.shape)
    