# Функция потерь
"""
TODO(ПЕРЕВЕСТИ ВСЕ В ООП)
"""

from torch import nn
from typing import Optional
import torch 
mse = torch.nn.MSELoss(reduction='none')
def gaussian_kernel(x, y, sigma=1):
    #TODO dim=1 надо заметить возможно на dim=0
    x_sq = torch.sum(x**2, dim=1, keepdim=True)
    y_sq = torch.sum(y**2, dim=1, keepdim=True)
    pairwise_dist = x_sq + y_sq.t() - 2 * torch.mm(x, y.t())
    return torch.exp(-pairwise_dist / (2 * sigma**2))
def compute_mmd(z, sigma=1.0):
    """
    z: латентные коды из энкодера [batch_size, latent_dim]
    reg_weight: коэффициент λ для MMD (начните с 10-100)
    sigma: параметр σ гауссова ядра
    """
    batch_size = z.size(0)
    
    # 1. Сэмплируем из априорного распределения p(z) = N(0, I)
    prior_z = torch.randn_like(z)
    
    # 2. Вычисляем попарные расстояния для ядра
    
    # 3. Вычисляем три терма MMD
    zz = gaussian_kernel(z, z, sigma=sigma)
    pp = gaussian_kernel(prior_z, prior_z, sigma=sigma)
    zp = gaussian_kernel(z, prior_z, sigma=sigma)
    
    # 4. Исключаем диагональные элементы (i=j)
    mmd = (zz.sum() - batch_size)/ (batch_size * (batch_size - 1)) + \
          (pp.sum() - batch_size)/ (batch_size * (batch_size - 1)) - \
          2 * zp.sum() / (batch_size * batch_size)
    
    return mmd
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
        mask[:, 0, :] = 0  # начало (start token)
        # Определяем активные детекторы: хотя бы одна фича != mask_v
        detector_active = torch.sum(mask, dim=2) > 0  # (batch, det) - True если детектор активен
        batch = detector_active.shape[0]
        # Находим последний активный детектор (стоп-токен) для каждого события
        for b in range(batch):
            active_indices = torch.where(detector_active[b])[0]
            if len(active_indices) > 0:
                last_active_idx = active_indices[-1]
                mask[b, last_active_idx, :] = 0  # зануляем стоп-токен
    
    if use_mask:
        return mask
    else:
        # Возвращаем количество активных детекторов (детектор активен, если хотя бы одна фича != mask_v)
        detector_active = torch.sum(mask, dim=2) > 0  # (batch, det)
        return torch.sum(detector_active, dim=1)  # (batch,)
def Num_Det_Loss(lengths_real: torch.Tensor, lenght_fake: torch.Tensor, reduction: str = 'none'):
    """MSE (target − pred)² по числу детекторов на событие, форма (batch,).

    Важно: (B,) и (B, 1) без выравнивания в MSELoss дают broadcast (B, B) — неверный лосс.
    """
    lr = lengths_real.float().reshape(-1)
    lf = lenght_fake.float().reshape(-1)
    if lr.shape[0] != lf.shape[0]:
        raise ValueError(f"Num_Det_Loss: размер батча не совпадает: {lr.shape[0]} vs {lf.shape[0]}")
    se = (lr - lf) ** 2
    if reduction == 'mean':
        return se.mean()
    return se
def CE_loss_particle(pred : torch.Tensor, real : torch.Tensor):
    # assert pred.shape = real.shape
    loss = nn.CrossEntropyLoss(reduction='none')(pred, real)
    return torch.mean(loss)
def vae_loss(recon_x, x, mu, log_var, pred_num, recon_pred, params_CR, real_part, mask = -10.0, use_mask: bool = True, koef_loss: Optional[torch.Tensor] = None,
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
    #MMD for INFO VAE
    mmd_loss = compute_mmd(mu)
    

    # КОэйфиценты лосса
    if koef_loss is None:
        koef_loss = torch.ones(1,6)
    koef_loss = koef_loss.unsqueeze(0)
    try:
        recon_loss = nn.MSELoss(reduction='none')(recon_x, x) # return shape - batch, det, featch
    except Exception as e:
        print(recon_x.shape, x.shape, e)
        raise e
    recon_loss *= koef_loss
    try:
        kl_elem = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    except TypeError:
        kl_elem = torch.tensor(0.0, device=x.device)
    if kl_elem.dim() == 0:
        kl_scalar = kl_elem
        kl_per_event = torch.zeros(x.size(0), device=x.device, dtype=kl_scalar.dtype)
    else:
        # Сначала сумма по батчу, затем среднее по фичам латента (как вы просили)
        kl_scalar = kl_elem.sum(dim=0).mean()
        # На событие: сумма по латенту (стандартный KL на один x)
        kl_per_event = kl_elem.sum(dim=1)
    # подсчет кол-ва детекторов в событии
    # mask = calc_det(x, mask)
    if use_mask:
        # Усредняем по активным детекторам (Так и надо)

        num_det_mask = calc_det(x, mask, use_mask=use_mask)
        recon_loss*=num_det_mask
        # Правильно считаем количество активных детекторов: детектор активен, если хотя бы одна фича != mask
        detector_active = torch.sum(num_det_mask, dim=2) > 0  # (batch, det)
        num_det = torch.sum(detector_active, dim=1, keepdim=True).unsqueeze(-1).float()  # (batch, 1, 1)
        recon_loss = torch.sum(recon_loss/num_det, dim=1).mean(dim=1) # mean by active det
        num_det_loss = Num_Det_Loss(num_det.squeeze().float(), pred_num)
        if not(reduce_loss_per_event):
            recon_loss = torch.mean(recon_loss) # mean by batch and featches
            num_det_loss = torch.mean(num_det_loss)
        # loss for predict num active detections
        
        # loss for predict particles
        if recon_pred is not None:
            loss_recon_pred = nn.MSELoss(reduction='none')(recon_pred, params_CR.float()) #batch, params
            if reduce_loss_per_event:
                loss_recon_pred = torch.mean(loss_recon_pred, dim=1) # mean by params, shape: (batch,)
            else:
                loss_recon_pred = torch.mean(loss_recon_pred, dim=0) # mean by batch, shape: (params,)
            # CE_loss_particle(pred_part, real_part)
        else:
            if reduce_loss_per_event:
                loss_recon_pred = torch.zeros(x.size(0), device=x.device)
            else:
                loss_recon_pred = torch.zeros(1,)
        
        if reduce_loss_per_event:
            return recon_loss, kl_per_event, num_det_loss, loss_recon_pred, mmd_loss
        else:
            return recon_loss, kl_scalar, num_det_loss, loss_recon_pred, mmd_loss
    else:
        recon_loss = torch.mean(recon_loss) # mean by active det
        
        # Вычисляем num_det_loss даже при use_mask=False
        num_det = calc_det(x, mask, use_mask=False) # shape: (batch,)
        num_det_loss = Num_Det_Loss(num_det.float(), pred_num)
        if not(reduce_loss_per_event):
            num_det_loss = torch.mean(num_det_loss)
        
        # loss for predict particles
        if recon_pred is not None:
            loss_recon_pred = nn.MSELoss(reduction='none')(recon_pred, params_CR.float()) #batch, params
            if reduce_loss_per_event:
                loss_recon_pred = torch.mean(loss_recon_pred, dim=1) # mean by params, shape: (batch,)
            else:
                loss_recon_pred = torch.mean(loss_recon_pred, dim=0) # mean by batch, shape: (params,)
        else:
            if reduce_loss_per_event:
                loss_recon_pred = torch.zeros(x.size(0), device=x.device)
            else:
                loss_recon_pred = torch.zeros(1,)
        
        if reduce_loss_per_event:
            return recon_loss, kl_per_event, num_det_loss, loss_recon_pred, mmd_loss
        return recon_loss, kl_scalar, num_det_loss, loss_recon_pred, mmd_loss

def vae_loss_none(recon_x, x, mu, log_var, pred_num, pred_part=None, real_part=None, recon_pred=None, params_CR=None, mask = -10.0, use_mask: bool = True, koef_loss: Optional[torch.Tensor] = None):
    """
    Выдает лосс без усреднения (recon_loss остаётся (batch, det, feat)).
    Для infoVAE: передать recon_pred и params_CR (MSE по параметрам реконструкции).
    Для модели с классификатором частиц: передать pred_part и real_part (CE).
    Всегда возвращает 5 значений, включая mmd_loss для совместимости с vae_loss.
    """
    mmd_loss = compute_mmd(mu)
    if koef_loss is None:
        koef_loss = torch.ones(1,6)
    koef_loss = koef_loss.unsqueeze(0)
    recon_loss = nn.MSELoss(reduction='none')(recon_x, x) # return shape - batch, det, featch
    recon_loss *= koef_loss
    kl_elem = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kl_agg = kl_elem.sum(dim=0).mean() if kl_elem.dim() > 0 else kl_elem
    # подсчет кол-ва детекторов в событии
    if use_mask:
        num_det_mask = calc_det(x, mask, use_mask=use_mask)
        recon_loss*=num_det_mask
        detector_active = torch.sum(num_det_mask, dim=2) > 0  # (batch, det)
        num_det = torch.sum(detector_active, dim=1).float()  # (batch,)
        num_det_loss = Num_Det_Loss(num_det, pred_num)
        # infoVAE: loss по реконструкции параметров; иначе CE по типу частиц
        if recon_pred is not None and params_CR is not None:
            loss_aux = nn.MSELoss(reduction='none')(recon_pred, params_CR.float())
            loss_aux = torch.mean(loss_aux, dim=1)  # (batch,)
        elif pred_part is not None and real_part is not None:
            loss_aux = CE_loss_particle(pred_part, real_part)
        else:
            loss_aux = torch.zeros(x.size(0), device=x.device)
        return recon_loss, kl_agg, num_det_loss, loss_aux, mmd_loss
    else:
        num_det = calc_det(x, mask, use_mask=False)
        num_det_loss = Num_Det_Loss(num_det.float(), pred_num)
        if recon_pred is not None and params_CR is not None:
            loss_aux = nn.MSELoss(reduction='none')(recon_pred, params_CR.float()).mean(dim=1)
        elif pred_part is not None and real_part is not None:
            loss_aux = CE_loss_particle(pred_part, real_part)
        else:
            loss_aux = torch.zeros(x.size(0), device=x.device)
        return recon_loss, kl_agg, num_det_loss, loss_aux, mmd_loss
if __name__ == "__main__":
    input = torch.randn(3, 5,6, requires_grad=True)
    target = torch.randn(3, 5,6,)
    output = mse(input, target)
    print(output.shape)

