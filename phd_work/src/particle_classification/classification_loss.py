# Функция потерь
from torch import nn
from typing import Optional
import torch 
import numpy as np
mse = torch.nn.MSELoss(reduction='mean')

class ClassificationLoss(nn.Module):
    def __init__(self, num_class: int = 2):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.num_class = num_class
    def forward(self, pred: torch.Tensor, real: torch.Tensor):
        '''
        real has shape (b,)
        pred has shape (b, num_class)
        '''
        loss = self.loss_fn(pred, real)
        return torch.mean(loss)
if __name__ == "__main__":
    pred = torch.randn(3, 2)
    real = torch.tensor([1, 0, 1])
    print(pred, real)
    loss = nn.CrossEntropyLoss(reduction='none')(pred, real)
    print(loss)
    loss = ClassificationLoss()(pred, real)
    print(loss)