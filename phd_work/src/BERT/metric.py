
from torch import nn
from typing import Optional
import torch 
import sklearn
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

class DivedeMetrics():
    def __init__(self, TBwriter = None, num_class: int = 2):
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
    def plot_hist(self, loss_pr, loss_ph, ax):
        ax.hist(loss_pr, label= 'proton', histtype = 'step',  log=True, density =True)
        ax.hist(loss_ph, label= 'photon', histtype = 'step',  log=True, density =True)
        ax.set_title("Loss hist")
        ax.set_xlabel('loss')
        ax.set_ylabel('log(num)')
        ax.legend()
        return ax

    def plot_roc_curve(self, loss_pr, loss_ph, ax):
        """
        Строит ROC кривую и вычисляет AUC
        
        Args:
            y_true: истинные метки (0 или 1)
            y_scores: вероятности положительного класса (второй столбец из softmax)
            epoch: номер эпохи для логирования
        """
        label_pr = np.zeros_like(loss_pr)
        label_ph = np.ones_like(loss_ph)
        label = np.concatenate((label_pr, label_ph))
        loss = np.concatenate((loss_pr, loss_ph))
        # Вычисляем ROC кривую
        fpr, tpr, thresholds = roc_curve(label, loss)
        roc_auc = auc(fpr, tpr)
        
        # Создаем график
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier (AUC = 0.5)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        return ax
    def plot_tpr_vs_fpr(self, loss_pr, loss_ph, ax):
        """
        Строит графики TPR и FPR как функции порога (threshold по loss).

        Args:
            loss_pr: массив значений loss для класса proton (метка 0)
            loss_ph: массив значений loss для класса photon (метка 1)
            ax: matplotlib Axes
        """
        # метки и значения
        label_pr = np.zeros_like(loss_pr)
        label_ph = np.ones_like(loss_ph)
        y_true = np.concatenate((label_pr, label_ph))
        y_score = np.concatenate((loss_pr, loss_ph))

        # roc_curve вернёт fpr, tpr и thresholds
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        len_ph = label_ph.shape[0]
        len_pr = label_pr.shape[0]
        fp = fpr * len_pr
        tp = tpr * len_ph
        # Рисуем кривые
        ax.plot(thresholds, np.log10(tpr) - np.log10(fpr), color="green", lw=2, label="TPR/FPR vs threshold")

        ax.set_xlabel("Threshold (loss)")
        ax.set_ylabel("log10(TPR/FPR) Rate")
        ax.set_title("TPR/FPR as functions of threshold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        return ax
    def plot_tpr_fpr_vs_threshold(self, loss_pr, loss_ph, ax):
        """
        Строит графики TPR и FPR как функции порога (threshold по loss).

        Args:
            loss_pr: массив значений loss для класса proton (метка 0)
            loss_ph: массив значений loss для класса photon (метка 1)
            ax: matplotlib Axes
        """
        # метки и значения
        label_pr = np.zeros_like(loss_pr)
        label_ph = np.ones_like(loss_ph)
        y_true = np.concatenate((label_pr, label_ph))
        y_score = np.concatenate((loss_pr, loss_ph))

        # roc_curve вернёт fpr, tpr и thresholds
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        # Рисуем кривые
        ax.plot(thresholds, np.log10(tpr), color="green", lw=2, label="TPR vs threshold")
        ax.plot(thresholds, np.log10(fpr), color="red", lw=2, label="FPR vs threshold")

        ax.set_xlabel("Threshold (loss)")
        ax.set_ylabel("log10(Rate)")
        ax.set_title("TPR/FPR as functions of threshold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        return ax
    def __call__(self, loss_pr, loss_ph,
                ep=None, prob=None, tag: Optional[str] = None):
        fig, axs = plt.subplots(2,2,figsize = (12,12))
        axs[0,0] = self.plot_roc_curve(loss_pr, loss_ph, axs[0,0])
        axs[0,1] = self.plot_hist(loss_pr, loss_ph, axs[0,1])
        axs[1,0] = self.plot_tpr_vs_fpr(loss_pr, loss_ph, axs[1,0])
        axs[1,1] = self.plot_tpr_fpr_vs_threshold(loss_pr, loss_ph, axs[1,1])
        if self.TBwriter:
            name = "val/plots"
            if tag:
                name += f"/{tag}"
            if prob:
                name += f"_prob_{str(prob)}"
            self.TBwriter.add_figure(name, fig, ep)
if __name__ == '__main__':
    metric = DivedeMetrics()
    loss_pr = np.random.rand(10)
    loss_ph = np.random.rand(10)
    metric(loss_pr, loss_ph)
        

        