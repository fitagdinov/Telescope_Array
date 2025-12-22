import umap.umap_ as umap
import numpy as np
from typing import Optional, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

#Devided
from torch import nn
import torch 
import sklearn
from sklearn.metrics import roc_curve, auc, roc_auc_score
class LatentMetric():
    def __init__(self, num_split: Optional[int] = None):
        self.num_split = num_split

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
    def umap_fig(self, proton_st: np.array, photon_st: np.array, ax2=None):# List[plt.Axes, plt.Axes]
        all_st= np.concatenate([proton_st, photon_st], axis=0) 
        lable = np.array([0]*len(proton_st) + [1]*len(photon_st))
        reducer = umap.UMAP()
        manifold = reducer.fit(all_st)
        X_pr = manifold.transform(proton_st)
        X_ph = manifold.transform(photon_st)
        X_reduced = manifold.transform(all_st)
        names = ['proton', 'photon']
        # ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=lable, cmap='viridis', alpha=0.7, label=names)
        # ax.legend()
        if not(ax2 is None):
            ax_1,ax_2  = ax2
            ax_1.scatter(X_pr[:, 0], X_pr[:, 1], s=0.5, alpha=0.3, label=names[0])
            ax_2.scatter(X_ph[:, 0], X_ph[:, 1], s=0.5, alpha=0.3, label=names[1])
            ax2 = (ax_1,ax_2)
            ax_1.legend()
            ax_2.legend()

        return ax2
    def calc_classification_metric(self, proton_st: np.array, photon_st: np.array, ax: plt.Axes):
        X = np.vstack([proton_st, photon_st])
        y = np.array([0] * len(proton_st) + [1] * len(photon_st))
        # Разделение на тренировочную и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            stratify=y,
            random_state=42
        )

        # Создание и обучение CatBoost классификатора
        model = CatBoostClassifier(
            iterations=10000,
            learning_rate=0.1,
            depth=6,
            verbose=100,
            early_stopping_rounds=50,
            random_state=42
        )

        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            plot=True
        )

        # Предсказание на тестовой выборке
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Оценка модели
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        # ax.text(0.5, 0.5, classification_report(y_test, y_pred), fontsize=14, ha='center', va='center')
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax)
        print("\nConfusion Matrix:")
        print(y_test, y_pred)
        print(confusion_matrix(y_test, y_pred))
        f1_score_ = f1_score(y_test, y_pred)
        return f1_score_, ax
    
    def __call__(self, proton_st: np.array, photon_st: np.array) -> tuple[float, plt.Figure, plt.Axes]:
        if self.num_split is not None:
            np.random.seed(42)
            index = np.random.random_integers(0, len(photon_st)-1, int(self.num_split), )
            print(index)
            proton_st = proton_st[index]
            photon_st = photon_st[index]
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        # roc curve
        ax[1] = self.umap_fig(proton_st, photon_st, ax[1])
        f1_score_, ax[0,1] = self.calc_classification_metric(proton_st, photon_st, ax[0,1])
        return f1_score_, fig
class DivedeMetrics():
    def __init__(self, num_split: Optional[int] = None, num_class: int = 2):
        self.num_split = num_split
        # super(nn.Module).__init__()
        
        self.num_class = num_class
    def plot_hist(self, loss_pr, loss_ph, ax):
        ax.hist(loss_pr, label= 'proton', histtype = 'step',  log=True, density =True)
        ax.hist(loss_ph, label= 'photon', histtype = 'step',  log=True, density =True)
        ax.set_title("Loss hist")
        ax.set_xlabel('loss')
        ax.set_ylabel('log(num)')
        ax.legend()
        return ax

    def plot_roc_curve(self, loss_pr, loss_ph, ax, title: Optional[str] = None):
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
        if title is not None:
            ax.set_title(title + " Receiver Operating Characteristic (ROC) Curve")
        else:
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        return ax
    # def plot_tpr_vs_fpr(self, loss_pr, loss_ph, ax, title: Optional[str] = None):
    #     """
    #     Строит графики TPR и FPR как функции порога (threshold по loss).

    #     Args:
    #         loss_pr: массив значений loss для класса proton (метка 0)
    #         loss_ph: массив значений loss для класса photon (метка 1)
    #         ax: matplotlib Axes
    #     """
    #     # метки и значения
    #     label_pr = np.zeros_like(loss_pr)
    #     label_ph = np.ones_like(loss_ph)
    #     y_true = np.concatenate((label_pr, label_ph))
    #     y_score = np.concatenate((loss_pr, loss_ph))

    #     # roc_curve вернёт fpr, tpr и thresholds
    #     fpr, tpr, thresholds = roc_curve(y_true, y_score)
    #     len_ph = label_ph.shape[0]
    #     len_pr = label_pr.shape[0]
    #     fp = fpr * len_pr
    #     tp = tpr * len_ph
    #     # Рисуем кривые
    #     ax.plot(thresholds, np.log10(tpr) - np.log10(fpr), color="green", lw=2, label="TPR/FPR vs threshold")

    #     ax.set_xlabel("Threshold (loss)")
    #     ax.set_ylabel("log10(TPR/FPR) Rate")
    #     if title is not None:
    #         ax.set_title(title)
    #     else:
    #         ax.set_title("TPR/FPR as functions of threshold")
    #     ax.legend(loc="best")
    #     ax.grid(True, alpha=0.3)

    #     return ax
    def plot_tpr_fpr_vs_threshold(self, loss_pr, loss_ph, ax, title: Optional[str] = None):
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
        if title is not None:
            ax.set_title(title + "TPR/FPR as functions of threshold")
        else:
            ax.set_title("TPR/FPR as functions of threshold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        return ax
    def __call__(self, loss_pr, loss_ph,
                loss_pr_KL, loss_ph_KL,
                ):
        if self.num_split is not None:
            np.random.seed(42)
            index = np.random.random_integers(0, len(loss_ph)-1, int(self.num_split), )
            print(index)
            loss_pr = loss_pr[index]
            loss_ph = loss_ph[index]
            loss_pr_KL = loss_pr_KL[index]
            loss_ph_KL = loss_ph_KL[index]
        fig, axs = plt.subplots(2,2,figsize = (12,12))
        axs[0,0] = self.plot_roc_curve(loss_pr, loss_ph, axs[0,0], title="Reconstruction")
        axs[0,1] = self.plot_roc_curve(loss_pr_KL, loss_ph_KL, axs[0,1], title="KL")

        axs[1,0] = self.plot_tpr_fpr_vs_threshold(loss_pr, loss_ph, axs[1,0], title="Reconstruction")
        axs[1,1] = self.plot_tpr_fpr_vs_threshold(loss_pr_KL, loss_ph_KL, axs[1,1], title="KL")

        return None,fig
if __name__ == "__main__":
    proton_st = np.random.randn(100, 2)
    photon_st = np.random.randn(100, 2)
    metric = DivedeMetrics(num_split=3)
    metric_f1, fig = metric(proton_st, photon_st)
    print(metric_f1)
    plt.show()
