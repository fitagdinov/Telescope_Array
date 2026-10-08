
from torch import nn
from typing import Optional
import torch 
import sklearn
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

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
    
    def plot_roc_curve(self, y_true, y_scores, epoch=-1):
        """
        Строит ROC кривую и вычисляет AUC
        
        Args:
            y_true: истинные метки (0 или 1)
            y_scores: вероятности положительного класса (второй столбец из softmax)
            epoch: номер эпохи для логирования
        """
        # Вычисляем ROC кривую
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Создаем график
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier (AUC = 0.5)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Добавляем в TensorBoard
        if not(self.TBwriter is None):
            self.TBwriter.add_figure("ROC_curve", plt.gcf(), epoch)
            self.TBwriter.add_scalar("classification/ROC_AUC", roc_auc, epoch)
        
        plt.close()  # Закрываем фигуру для экономии памяти
        
        return roc_auc, fpr, tpr, thresholds
    def __call__(self, pred: torch.Tensor, real: torch.Tensor, epoch: int=-1, show: bool = False):
        '''
        real has shape (b,)
        pred has shape (b, num_class)
        '''
        pred_class = np.argmax(pred, axis=1)
        
        # Строим confusion matrix
        confusion_matrix = self.confusion_matrix(real, pred_class)
        cm_display = ConfusionMatrixDisplay(confusion_matrix).plot()
        if not(self.TBwriter is None):
            self.TBwriter.add_figure("confusion_matrix", cm_display.figure_, epoch)
        else:
            print('confusion_matrix', confusion_matrix)
        # Строим ROC кривую (только для бинарной классификации)
        if self.num_class == 2:
            # Берем вероятности положительного класса (второй столбец)
            y_scores = pred[:, 1]  # Вероятности класса 1 (proton)
            roc_auc, fpr, tpr, thresholds = self.plot_roc_curve(real, y_scores, epoch)
            
            if show:
                print(f"ROC AUC: {roc_auc:.4f}")
        
        # Вычисляем стандартные метрики
        res = {}
        for metric_name, metric_func in self.dictMetrics.items():
            metric_value = metric_func(real, pred_class)
            if not(self.TBwriter is None):
                self.TBwriter.add_scalar(f"classification/{metric_name}", metric_value, epoch)
            if show:
                print(f"{metric_name}: {metric_value}")
            if metric_name == 'f1_score':
                self.score = metric_value
            res[metric_name] = metric_value
        
        # Добавляем ROC AUC в результаты
        if self.num_class == 2:
            res['roc_auc'] = roc_auc
            
        return res
        