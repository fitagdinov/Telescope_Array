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

class ClassificationPipline(Pipline):
    def __init__(self, config, need_train_DS: bool = True,  many_val_loaders =False):
        super().__init__(config, need_train_DS,  many_val_loaders =False)
        self.device = 'cuda'
        if self.config['used_model'] == 'Simple_classifiacation_model':
            self.model = Class_models.Simple_classifiacation_model(
                                                               **self.config)
        elif self.config['used_model'] == 'TransformerClassificationModel':
            self.model = Class_models.TransformerClassificationModel(
                                                               **self.config)
        else:
            raise ValueError('Unknown model type')
        self.model.to(self.device)
        self.Loss = ClassificationLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.best_score = -1.0
        self.Metrics = ClassificationMetrics(TBwriter=self.writer, num_class = len(self.config['paticles']))
        self.save_model_path = os.path.join(self.config['save_model_path'], self.config['PATH'])
    def train(self):
        self.loss_best = 1000
        iters = 0
        for epoch in range(self.epochs):
            print('lr_scheduler', self.optimizer.param_groups[0]['lr'])
            self.writer.add_scalar("lr_scheduler", self.optimizer.param_groups[0]['lr'], epoch)
            self.model.train()
            pbar = tqdm(self.train_loader, desc =f"TRAIN Epoch {epoch + 1}/{self.epochs}, Loss: 0.0")
            for x, part, _ in pbar:  # x должен быть пакетом последовательностей с заполнением
                self.optimizer.zero_grad()
                x = x.to(self.device)
                part = torch.where(part == 1, 0, 1).to(self.device) # 0- photon, 1- proton
                pred_mass = self.model(x)
                loss = self.Loss(pred_mass, part)
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar("train/Loss", loss, iters)
                pbar.set_description(f"TRAIN Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")
                iters += 1
            
            self.validation(epoch=epoch, analys=True)
    def validation(self, epoch, analys=True):
        self.model.eval()
        loss_mean = []
        y_preds = None
        y_target = None
        with torch.no_grad():
            # self.val_loaders is list which has one dataloader if many_val_loaders = Flase in piplene
            for x, part, _ in tqdm(self.val_loaders[0]):
                x = x.to(self.device)
                part = torch.where(part == 1, 0, 1).to(self.device) # 0- photon, 1- proton
                pred_mass = self.model(x)
                loss = self.Loss(pred_mass, part)
                loss_mean.append(loss.item())
                if y_preds is None:
                    y_preds = pred_mass.detach().cpu().numpy()
                    y_target =  part.detach().cpu().numpy()
                else:
                    y_preds = np.concatenate([y_preds, pred_mass.detach().cpu().numpy()])
                    y_target = np.concatenate([y_target, part.detach().cpu().numpy()])
        self.Metrics(y_preds, y_target, epoch)
        if self.Metrics.score > self.best_score:
            self.best_score = self.Metrics.score
            torch.save(self.model.state_dict(), os.path.join(self.PATH, f'best'))
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
            for x, part, _ in tqdm(getting_dataloader):
                x = x.to(self.device)
                part = torch.where(part == 1, 0, 1).to(self.device) # 0- photon, 1- proton
                pred_mass = self.model(x)
                if y_preds is None:
                    y_preds = pred_mass.detach().cpu().numpy()
                    y_target =  part.detach().cpu().numpy()
                else:
                    y_preds = np.concatenate([y_preds, pred_mass.detach().cpu().numpy()])
                    y_target = np.concatenate([y_target, part.detach().cpu().numpy()])
        metric_res = self.Metrics(y_preds, y_target)
        print(metric_res)
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
        # pipline.train()
    # elif args.mode == 'test':
    #     pipline = Pipline(config, need_train_DS=False)
    #     print('TEST PIPLINE')
    #     pipline.validation()
    # elif args.mode == 'latent':
    #     pipline = Pipline(config, need_train_DS=False)
    #     print('Latent PIPLINE')
        # pipline.predict_latent(args.write_embading)