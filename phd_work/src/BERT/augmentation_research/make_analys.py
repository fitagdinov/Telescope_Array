import numpy as np
import os
import matplotlib.pyplot as plt
import sklearn
import umap.umap_ as umap
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook
# for classifcator 
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay, recall_score, precision_score

class Analys():
    def __init__(self, path:str, num_train:int = 20000):

        self.num_train = 20000
        self.path = path
        all_emb = {}
        for part in os.listdir(path):
            all_emb[part] = {}
            for f in os.listdir(path + '/' + part):
                file = path + '/' + part + '/' + f
                n = f.split('_')[-1].replace('.npy', '')
                n=int(n)
                all_emb[part][n] = np.load(file)
        self.all_emb = all_emb
        


    # lernong classification
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
            iterations=3000,
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
        # ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax)
        # print("\nConfusion Matrix:")
        # print(y_test, y_pred)
        # print(confusion_matrix(y_test, y_pred))
        f1_score_ = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        return f1_score_,recall, precision, ax, model 
    
    def learning_classificator(self):
        all_emb = self.all_emb
        num_train = self.num_train
        Classificators = []
        F1_list = []
        recall_list = []
        precision_list = []
        num_iter = len(all_emb['proton'].keys())
        # num_iter = 1
        # fig, axs = plt.subplots(num_iter,1, figsize=(5,6*num_iter), sharex=True)
        for i in tqdm_notebook(range(num_iter)):
            proton_st = all_emb['proton'][i][:num_train]
            photon_st = all_emb['photon'][i][:num_train]
            ax = None#axs[i]
            f1_score_,recall, precision, ax, model  = self.calc_classification_metric(proton_st, photon_st, ax)
            print(f'f1_score in iteration {i}: {f1_score_}')
            Classificators.append(model)
            F1_list.append(f1_score_)
            recall_list.append(recall)
            precision_list.append(precision)
        self.Classificators = Classificators
        self.F1_list = F1_list
        return Classificators, F1_list, recall_list, precision_list
    def plot_metrics(self):
        plt.plot(F1_list, label = 'F1')
        plt.plot(recall_list, label = 'Recall')
        plt.plot(precision_list, label = 'Precision')
        plt.title('Metrics. Learning each iteration\n 0-proton 1-photon')
        plt.xlabel('Iteration')
        plt.ylabel('Metric')
        plt.grid()
        plt.legend()