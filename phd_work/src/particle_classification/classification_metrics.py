
from torch import nn
from typing import Optional
import torch 
import sklearn
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

class ClassificationMetrics():
    def __init__(self, TBwriter, num_class: int):
        # super(nn.Module).__init__()
        self.num_class = num_class
        self.precision = sklearn.metrics.precision_score
        self.recall = sklearn.metrics.recall_score
        self.f1_score = sklearn.metrics.f1_score
        self.confusion_matrix = sklearn.metrics.confusion_matrix
        self.dictMetrics = {'precision': self.precision,
                            'recall': self.recall,
                            'f1_score': self.f1_score,}
        self.TBwriter = TBwriter
        self.score = None
    def __call__(self, pred: torch.Tensor, real: torch.Tensor, epoch: int, show: bool = False):
        '''
        real has shape (b,)
        pred has shape (b, num_class)
        '''
        pred_class = np.argmax(pred, axis =1)
        confusion_matrix = self.confusion_matrix(real, pred_class)
        cm_display = ConfusionMatrixDisplay(confusion_matrix).plot()
        self.TBwriter.add_figure("confusion_matrix", cm_display.figure_, epoch)
        for metric_name, metric_func in self.dictMetrics.items():
            metric_value = metric_func(real, pred_class)
            self.TBwriter.add_scalar(f"classification/{metric_name}", metric_value, epoch)
            if show:
                print(f"{metric_name}: {metric_value}")
            if metric_name == 'f1_score':
                self.score = metric_value
        