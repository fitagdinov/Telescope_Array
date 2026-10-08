# Функция потерь
from torch import nn
from typing import Optional
import torch 
import numpy as np
mse = torch.nn.MSELoss(reduction='mean')

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