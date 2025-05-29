# Функция потерь
"""
TODO(ПЕРЕВЕСТИ ВСЕ В ООП)
"""

from torch import nn
from typing import Optional
import torch 
mse = torch.nn.MSELoss(reduction='mean')


def calc_det(x: torch.Tensor, mask_v: float = -10.0, start_stop_teken: bool = True, use_mask: bool = True) -> torch.Tensor:
    """
    Вычисляет маску активных детекторов или число детекторов на событие.

    Аргументы:
        x (Tensor): Входной тензор (batch, det, feat).
        mask_v (float): Значение, используемое как маска (например, -10).
        start_stop_teken (bool): Если True, обнуляет start/stop токены.
        use_mask (bool): Если False — возвращает число детекторов на событие.

    Возвращает:
        Tensor: маска или числа детекторов.
    """
    # Определим маску: 1 там, где не маска, 0 — где паддинг
    mask = torch.where(x != mask_v, 1, 0)

    # Если нужно убрать start/stop токены
    if start_stop_teken:
        mask[:, 0] = 0  # начало
        num_det = torch.sum(mask, dim=1)
        batch = num_det.shape[0]
        # занулим стоп-токен
        mask[torch.arange(batch), num_det[:, 0], :] = 0
        num_det = torch.sum(mask, dim=1)  # пересчитаем
        return mask

    if use_mask:
        return mask
    else:
        # просто вернем количество активных детекторов
        return torch.sum(mask, dim=1)
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
    """
    Общая функция потерь для VAE: включает MSE, KL-дивергенцию, число детекторов и классификацию массы.

    Аргументы:
        recon_x (Tensor): Восстановленные данные (batch, det, feat).
        x (Tensor): Истинные данные.
        mu, log_var (Tensor): Параметры латентного пространства.
        pred_num (Tensor): Предсказанная длина.
        pred_part (Tensor): Логиты по типу частиц.
        real_part (Tensor): Истинные классы.
        mask (float): Значение маски.
        use_mask (bool): Учитывать ли маску.
        koef_loss (Tensor): Коэффициенты весов по фичам.
        reduce_loss_per_event (bool): Если True — возвращает loss на событие.

    Возвращает:
        Tuple: (recon_loss, kl_div, num_det_loss, mass_loss)
    """

    # КОэйфиценты лосса
    if koef_loss is None:
        koef_loss = torch.ones(1,6)
    koef_loss = koef_loss.unsqueeze(0)
    recon_loss = nn.MSELoss(reduction='none')(recon_x, x) # return shape - batch, det, featch
    recon_loss *= koef_loss
    try:
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    except TypeError:
        kl_divergence = torch.zeros_like(recon_loss)
    # подсчет кол-ва детекторов в событии
    # mask = calc_det(x, mask)
    if use_mask:
        # Усредняем по активным детекторам (Так и надо)
        num_det_mask = calc_det(x, mask, use_mask=use_mask)
        recon_loss*=num_det_mask
        num_det = torch.sum(num_det_mask, dim=1)[:,0][:,None, None]
        recon_loss = torch.sum(recon_loss/num_det, dim=1) # mean by active det
        if not(reduce_loss_per_event):
            recon_loss = torch.mean(recon_loss) # mean by batch and featches
        # loss for predict num active detections
        num_det_loss = Num_Det_Loss(num_det[:,:,0].float(), pred_num)
        # loss for predict particles
        loss_mass = CE_loss_particle(pred_part, real_part)
        return recon_loss, kl_divergence / x.size(0), num_det_loss, loss_mass
    else:
        # В тупую усредняем
        recon_loss = torch.mean(recon_loss) # mean by active det
        return recon_loss, kl_divergence / x.size(0), num_det_loss, loss_mass

def vae_loss_none(recon_x, x, mu, log_var, pred_num, pred_part, real_part, mask = -10.0, use_mask: bool = True, koef_loss: Optional[torch.Tensor] = None):
    """
    Выдает лосс без усреднения. Получается его нельзя вести как тензор. Длины разные
    """
    if koef_loss is None:
        koef_loss = torch.ones(1,6)
    koef_loss = koef_loss.unsqueeze(0)
    recon_loss = nn.MSELoss(reduction='none')(recon_x, x) # return shape - batch, det, featch
    recon_loss *= koef_loss
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # подсчет кол-ва детекторов в событии
    if use_mask:
        num_det_mask = calc_det(x, mask, use_mask=use_mask)
        recon_loss*=num_det_mask
        num_det = torch.sum(num_det_mask, dim=1)[:,0][:,None, None]
        num_det_loss = Num_Det_Loss(num_det[:,:,0].float(), pred_num)
        # loss for predict particles
        loss_mass = CE_loss_particle(pred_part, real_part)
        return recon_loss, kl_divergence / x.size(0), num_det_loss, loss_mass

