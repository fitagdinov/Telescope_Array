"""
Замениять в данных значения сигнала и времен на латеное значение всего события
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../train_VAE')))
import torch.nn as nn
from model import Encoder, Encoder_Transformer_AE, Encoder_Transformer
import torch 
import numpy as np

class ClassificationAddLattent(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, 
        embading_path = '/home/rfit/Telescope_Array/phd_work/Models/AutoEncoder/info_Transfoemr_MMD_0.05_KL_0.01/best',
        device:str = 'cuda:0'):
        super(ClassificationAddLattent, self).__init__()
        VAE_state_dict = torch.load(embading_path)
        
        padding_value = -11.0
        stop_token = -10.0
        self.encoder_model = Encoder_Transformer(6,64, 8,
    
                    stop_token = stop_token,
                    padding_value=padding_value)
        embading_state_dict = {}
        for k,v in VAE_state_dict.items():
            if 'encoder' in k:
                newk = k.replace('encoder.', '')
                # for transformer
                if 'last_' in k:
                    newk = newk.replace('last_', 'last_encoder.')
                embading_state_dict[newk] = v
        self.device = device
        self.encoder_model.load_state_dict(embading_state_dict)
        self.encoder_model.to(device)
        self.encoder_model.eval()
    def calc_det(self, x: torch.Tensor, mask_v: float = -10.0, start_stop_teken: bool = True, use_mask: bool = True) -> torch.Tensor:
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
    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        emb,_,_ = self.encoder_model(x)
        num_det = self.calc_det(x, mask_v=-11.0, start_stop_teken=True, use_mask=False)  # (batch,)
        # Нельзя repeat(1, seq, 1) на (B,1): для 2D тензора PyTorch даёт форму (1, B*seq, 1).
        #num_det = num_det.view(-1, 1, 1).expand(-1, x.shape[1], 1)  # (batch, seq, 1)
        #emb = emb.unsqueeze(1).expand(-1, 1, -1)  # (batch, seq, latent_dim)
        #emb = emb
        #x = torch.cat((num_det, emb), dim=2)
        num_det = num_det.unsqueeze(1)
        params = params.to(self.device)
        x = torch.cat((params, emb), dim=1)
        return x
if __name__ == '__main__':
    model = ClassificationAddLattent(input_dim=6, hidden_dim=64, latent_dim=8,
        embading_path = '/home/rfit/Telescope_Array/phd_work/Models/AutoEncoder/info_Transfoemr_MMD_0.05_KL_0.01/best',
        device='cuda:0')
    x = torch.randn(3, 5, 6)
    x = model(x)

    print(x)
    print(x.shape)