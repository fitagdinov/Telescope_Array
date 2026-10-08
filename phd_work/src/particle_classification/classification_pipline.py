from tqdm import tqdm

import sys
import os

# Append the parent directory of the current directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../train_VAE')))
import model as Model

import argparse
import h5py as h5
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
import model as Model
import datasets as DataSet
# from train_VAE import loss as Loss
from typing import Optional, Tuple, Union
import pytorch_warmup as warmup
from  torch.optim.lr_scheduler import ExponentialLR

from torch.utils.tensorboard import SummaryWriter
import yaml
import time

# from ..train_VAE.utils import get_time, get_params_str, show_pred, read_config, clean_mask
import logging
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

import classification_models as Class_models
from pipline import Pipline
from classification_loss import ClassificationLoss
from classification_metrics import ClassificationMetrics
logger = logging.getLogger()


ADD_LATTENT = True

class ClassificationPipline(Pipline):
    def __init__(self, config, need_train_DS: bool = True,  many_val_loaders =False):
        super().__init__(config, need_train_DS,  many_val_loaders =False)
        self.device = 'cuda'
        if ADD_LATTENT:
            from classification_add_lattent import ClassificationAddLattent
            self.model_encoder = ClassificationAddLattent(input_dim=6, hidden_dim=64, latent_dim=8,
                embading_path = '/home/rfit/Telescope_Array/phd_work/Models/AutoEncoder/info_Transfoemr_MMD_0.05_KL_0.01_new_MMD2_MMD_increase_cont/best',
                device='cuda:0')
        else:
            self.model_encoder = None
        if self.config['used_model'] == 'Simple_classifiacation_model':
            self.model = Class_models.Simple_classifiacation_model(
                                                               **self.config)
        elif self.config['used_model'] == 'TransformerClassificationModel':
            self.model = Class_models.TransformerClassificationModel(
                                                               **self.config)
        elif self.config['used_model'] == 'FullyConnectedClassificationModel':
            self.model = Class_models.FullyConnectedClassificationModel(
                                                               **self.config)
        else:
            raise ValueError('Unknown model type')
        self.model.to(self.device)
        # Вычисляем веса классов для борьбы с дисбалансом
        class_weights = self._calculate_class_weights()
        self.Loss = ClassificationLoss(
            num_class=len(self.config['paticles']['train']),
            class_weights=class_weights,
            label_smoothing=0.1,  # Добавляем label smoothing
            weight_decay=1e-4     # Добавляем L2 регуляризацию
        )
        print(self.model.parameters())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config['lr']))
        self.best_score = -1.0
        # SummaryWriter создаётся в базовом Pipline, здесь просто пробрасываем
        self.Metrics = ClassificationMetrics(
            TBwriter=self.writer,
            num_class=len(self.config['paticles']['train'])
        )
        self.save_model_path = os.path.join(self.config['save_model_path'], self.config['PATH'])
    
    def _calculate_class_weights(self):
        """Вычисляет веса классов для борьбы с дисбалансом"""
        print("Вычисляем веса классов...")
        
        # Подсчитываем количество примеров каждого класса
        class_counts = {}
        total_samples = 0
        
        for x, part, params, recos in self.train_loader:
            part = torch.where(part == 1, 0, 1)  # 0- photon, 1- proton
            unique, counts = torch.unique(part, return_counts=True)
            
            for cls, count in zip(unique, counts):
                cls = cls.item()
                if cls not in class_counts:
                    class_counts[cls] = 0
                class_counts[cls] += count.item()
                total_samples += count.item()
        
        print(f"Распределение классов: {class_counts}")
        
        # Вычисляем веса (обратно пропорционально частоте)
        num_classes = len(class_counts)
        class_weights = torch.zeros(num_classes)
        
        for cls, count in class_counts.items():
            class_weights[cls] = total_samples / (num_classes * count)
        
        print(f"Веса классов: {class_weights}")
        return class_weights.to(self.device)
    
    def train(self):
        self.loss_best = 1000
        iters = 0
        for epoch in range(self.epochs):
            print('lr_scheduler', self.optimizer.param_groups[0]['lr'])
            self.writer.add_scalar("lr_scheduler", self.optimizer.param_groups[0]['lr'], epoch)
            self.model.train()
            pbar = tqdm(self.train_loader, desc =f"TRAIN Epoch {epoch + 1}/{self.epochs}, Loss: 0.0")
            for x, part, params, recos in pbar:  # x должен быть пакетом последовательностей с заполнением
                self.optimizer.zero_grad()
                if self.model_encoder is not None:
                    x = self.model_encoder(x, recos)
                x = x.to(self.device)
                part = torch.where(part == 1, 0, 1).to(self.device) # 0- photon, 1- proton
                pred_mass = self.model(x)
                loss = self.Loss(pred_mass, part, list(self.model.parameters()))
                loss.backward()
                
                # Анализируем предсказания для диагностики
                # ЗАКОМеНТИРОВАТЬ ЕСЛИ МЕШАЕТСЯ
                with torch.no_grad():
                    predicted_classes = torch.argmax(pred_mass, dim=1)
                    class0_count = torch.sum(predicted_classes == 0).item()
                    class1_count = torch.sum(predicted_classes == 1).item()
                    confidence = torch.max(pred_mass, dim=1)[0].mean().item()
                    
                    if iters % 300 == 0:  # Выводим каждые 300 итераций
                        print(f"Iter {iters}: Predicted - Class 0: {class0_count}, Class 1: {class1_count}, Avg confidence: {confidence:.3f}")
                
                self.optimizer.step()
                self.writer.add_scalar("train/Loss", loss, iters)
                self.writer.add_scalar("train/Class0_predictions", class0_count, iters)
                self.writer.add_scalar("train/Class1_predictions", class1_count, iters)
                self.writer.add_scalar("train/Avg_confidence", confidence, iters)
                pbar.set_description(f"TRAIN Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")
                iters += 1
            
            self.validation(epoch=epoch, analys=True)
    def validation(self, epoch, analys=True, return_metric=False):
        self.model.eval()
        loss_mean = []
        y_preds = None
        y_target = None
        with torch.no_grad():
            # self.val_loaders is list which has one dataloader if many_val_loaders = Flase in piplene
            for x, part, params, recos in tqdm(self.val_loaders[0]):
                x = x.to(self.device)
                if self.model_encoder is not None:
                    x = self.model_encoder(x, recos)
                part = torch.where(part == 1, 0, 1).to(self.device) # 0- photon, 1- proton
                pred_mass = self.model(x)
                loss = self.Loss(pred_mass, part, list(self.model.parameters()))
                loss_mean.append(loss.item())
                if y_preds is None:
                    y_preds = pred_mass.detach().cpu().numpy()
                    y_target =  part.detach().cpu().numpy()
                else:
                    y_preds = np.concatenate([y_preds, pred_mass.detach().cpu().numpy()])
                    y_target = np.concatenate([y_target, part.detach().cpu().numpy()])
        # Вычисляем метрики с ROC кривой
        metrics_results = self.Metrics(y_preds, y_target, epoch, show=True)
        if return_metric:
            return metrics_results
        # Выводим ROC AUC в консоль
        if 'roc_auc' in metrics_results:
            print(f"ROC AUC: {metrics_results['roc_auc']:.4f}")
        
        # Сохраняем лучшую модель по F1 score или ROC AUC
        current_score = metrics_results.get('roc_auc', self.Metrics.score)
        if current_score > self.best_score:
            self.best_score = current_score
            torch.save(self.model.state_dict(), os.path.join(self.PATH, f'best'))
            print(f"Новая лучшая модель сохранена! ROC AUC: {current_score:.4f}")
        
        self.writer.add_scalar("val/Loss", np.mean(loss_mean), epoch)
    def test(self, chpt:str, getting_dataloader = None):
        self.model.load(chpt)
        self.model.eval()
        y_preds = None
        y_target = None
        if getting_dataloader is None:
            getting_dataloader = self.val_loaders[0]
        print(getting_dataloader)
        with torch.no_grad():
            # self.val_loaders is list which has one dataloader if many_val_loaders = Flase in piplene
            for x, part, params, recos in tqdm(getting_dataloader):
                x = x.to(self.device)
                if self.model_encoder is not None:
                    x = self.model_encoder(x, recos)
                part = torch.where(part == 1, 0, 1).to(self.device) # 0- photon, 1- proton
                pred_mass = self.model(x)
                if y_preds is None:
                    y_preds = pred_mass.detach().cpu().numpy()
                    y_target =  part.detach().cpu().numpy()
                else:
                    y_preds = np.concatenate([y_preds, pred_mass.detach().cpu().numpy()])
                    y_target = np.concatenate([y_target, part.detach().cpu().numpy()])
        metric_res = self.Metrics(y_preds, y_target, show=True)
        print("=== Результаты тестирования ===")
        for metric_name, value in metric_res.items():
            print(f"{metric_name}: {value:.4f}")
        return y_preds, y_target

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="A simple example of argparse")

    # Add optional arguments
    parser.add_argument("-m", "--mode", type=str, help="The output file to save results", default="train")
    parser.add_argument("-e", "--write_embading", type=bool, help="Write latent data in TB for project analys", default="True")
    # Parse the arguments
    args = parser.parse_args()

    config = 'classification_config.yaml'
    if args.mode == 'train':
        pipline = ClassificationPipline(config, many_val_loaders =False)
        print('TRAIN PIPLINE')
        pipline.train()
    elif args.mode == 'test':
        pipline = ClassificationPipline(config, many_val_loaders =False)
        print('TEST PIPLINE')
        path = '/home/rfit/Telescope_Array/phd_work/Models/Classification/test_particles/One_working_V2/best'
        pipline.model.load(path)
        metric = pipline.validation('epoch', return_metric = True)
        print(metric)