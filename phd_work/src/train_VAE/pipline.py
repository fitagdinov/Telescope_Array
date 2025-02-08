from tqdm import tqdm
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
print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
import model as Model
import datasets as DataSet
import loss as Loss
from typing import Optional, Tuple, Union
import pytorch_warmup as warmup
from  torch.optim.lr_scheduler import ExponentialLR

from torch.utils.tensorboard import SummaryWriter
import yaml
import time

from utils import get_time, get_params_str, show_pred, read_config, clean_mask
import logging
import tensorflow as tf
import tensorboard as tb
torch.manual_seed(42)
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

logger = logging.getLogger()
class Pipline():
    def __init__(self, config):
        config = read_config()
        self.config = config
        # self.prepipline(config=config)
    def prepipline(self, config):
        name = config['PATH'].split('/')[-1]
        writer = SummaryWriter(log_dir=os.path.join('runs', name))
        writer.add_text('hparams',  str(config))
        dataset = DataSet.VariableLengthDataset(config['data_path'], 'train', mc_params=True, particles=config['used_particles'])
        val_dataset = DataSet.VariableLengthDataset(config['data_path'], 'test', mc_params=True, particles=config['used_particles'])
        kwargs = DataSet.get_params_mask(config)
        collate_fn = DataSet.wrapper_mask(DataSet.collate_fn_many_args, mc_params_return = False, **kwargs)
        train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
        start_token = kwargs['start_token'].to(device)
        model = Model.VAE(config['input_dim'], config['hidden_dim'], config['latent_dim'], lstm2=config['lstm2'], start_token=start_token).to(device)
        if config['chpt'] != 'None':
            model.load(config['chpt'])
        optimizer = optim.Adam(model.parameters(), lr=float(config['lr']))
        # write augmentes
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.writer = writer
        return {'train_loader': train_loader, 'val_loader': val_loader, 'model': model, 'optimizer': optimizer, 'writer': writer}
    def load_chpt(self, chpt_path: str):
        self.model.load(chpt_path)
    def pretrain(self):
        config = self.config
        PATH = config['PATH'] + get_time() + get_params_str(config)
        self.PATH = PATH
        config['PATH'] = PATH
        print("Saving Path: {}".format(PATH))
        prepipline_dict = self.prepipline(config)
        self.train_loader = prepipline_dict['train_loader']
        self.val_loader = prepipline_dict['val_loader']
        self.model = prepipline_dict['model']
        self.optimizer = prepipline_dict['optimizer']
        self.writer = prepipline_dict['writer']
        self.epochs = config['epoches']
        self.mask = config['padding_value']
        self.show_index = config['show_index']
        self.koef_KL = config['koef_KL']
        self.koef_DL = config['koef_DL']

        self.use_mask = config['use_mask']
        self.stop_token = config['stop_token']
        self.start_token = config['start_token']
        koef_loss = torch.tensor(config['koef_loss']).unsqueeze(0).to('cpu')
        self.koef_loss = koef_loss.to(device)
        os.makedirs(PATH, exist_ok = True)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.3, patience=8, threshold=0.005,)
    def train(self):
        self.pretrain()
        self.loss_best = 1000
        iters = 0
        for epoch in range(self.epochs):
            print('lr_scheduler', self.optimizer.param_groups[0]['lr'])
            self.writer.add_scalar("lr_scheduler", self.optimizer.param_groups[0]['lr'], epoch)
            self.model.train()
            pbar = tqdm(self.train_loader, desc =f"TRAIN Epoch {epoch + 1}/{self.epochs}, Loss: 0.0")
            for x, par in pbar:  # x должен быть пакетом последовательностей с заполнением
                x = x.to(device)
                self.optimizer.zero_grad()
                recon_x, mu, log_var, pred_num = self.model(x)
                recon_loss, kl_divergence, num_det_loss = Loss.vae_loss(recon_x, x, mu, log_var, pred_num, mask=self.mask,
                                                                        use_mask=self.use_mask, koef_loss=self.koef_loss)
                num_det_loss *= self.koef_DL
                kl_divergence *= self.koef_KL
                loss = recon_loss + kl_divergence + num_det_loss
                self.writer.add_scalar("train/Loss", loss, iters)
                self.writer.add_scalar("train/KL_loss", kl_divergence, iters)
                self.writer.add_scalar("train/recon_loss", recon_loss, iters)
                self.writer.add_scalar("train/num_det_loss", num_det_loss, iters)
                loss.backward()
                self.optimizer.step()
                pbar.set_description(f"TRAIN Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")
                iters += 1
            
            self.validation(epoch=epoch)
    def validation(self, epoch: int) -> None:
        """
        Performs validation on the model using the validation dataset.

        Parameters:
        epoch (int): The current epoch number.

        Returns:
        None. The function updates the model's state_dict if the validation loss is the best so far.
        It also prints the validation loss and loss_best, updates the learning rate scheduler,
        and writes the validation metrics to TensorBoard.
        """
        model = self.model
        val_loader = self.val_loader
        epochs = self.epochs
        koef_KL = self.koef_KL
        koef_DL = self.koef_DL

        model.eval()
        loss_mean = []
        KL_loss_mean = []
        recon_loss_mean = []
        num_det_loss_mean = []
        det_metric_mean = []
        pbar_val = tqdm(val_loader, desc =f"VAL Epoch {epoch + 1}/{epochs}, Loss: 0.0")
        for x, par in pbar_val:  # x should be a batch of sequences with padding
            x = x.to(device)
            recon_x, mu, log_var, pred_num = model(x)
            recon_loss, kl_divergence, num_det_loss, det_metric = Loss.vae_loss(recon_x, x, mu, log_var, pred_num, mask=self.mask, use_mask = self.config['use_mask'], koef_loss=self.koef_loss, get_det_metric=self.config['get_det_metric'] )
            kl_divergence *= koef_KL
            num_det_loss *= koef_DL
            loss = recon_loss + kl_divergence + num_det_loss
            loss_mean.append(loss.item())
            KL_loss_mean.append(kl_divergence.item())
            recon_loss_mean.append(recon_loss.item())
            num_det_loss_mean.append(num_det_loss.item())
            det_metric_mean.append(det_metric[:,0].to('cpu'))
            pbar_val.set_description(f"VAL Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        loss_final = np.array(loss_mean).mean()
        if loss_final<self.loss_best:
            self.loss_best = loss_final
            torch.save(model.state_dict(), os.path.join(self.PATH, f'best'))
        print(f'Epoch {epoch + 1}, Loss: {loss_final} loss_best {self.loss_best}')
        if epoch>0:
            self.scheduler.step(loss_final)
        # write in TB
        self.writer.add_scalar("val/Loss", loss_final, epoch)
        self.writer.add_scalar("val/KL_loss", np.array(KL_loss_mean).mean(), epoch)
        self.writer.add_scalar("val/recon_loss", np.array(recon_loss_mean).mean(), epoch)
        self.writer.add_scalar("val/num_det_loss", np.array(num_det_loss_mean).mean(), epoch)
        self.writer.add_scalar("val/det_metric_mean", 1 - np.concatenate(det_metric_mean).mean(), epoch)

        #show from last batch
        real = x[self.show_index]
        fake = recon_x[self.show_index]
        num = pred_num[self.show_index]
        for ii in range(len(self.show_index)):
            # get from back side
            i = -ii
            fig = show_pred(real[i], fake[i], tokens = (self.start_token, self.stop_token, self.mask), lenght_predict = num[i])
            self.writer.add_figure(f"val/show_pred_{ii}", fig, epoch)
    def predict_latent(self):
        # TODO сделать в отдельной функции getl loader
        self.pretrain()
        config = self.config
        writer = SummaryWriter(log_dir=os.path.join('runs', 'tests'))
        test_dataset = DataSet.VariableLengthDataset(config['data_path'], 'test', mc_params=True)
        kwargs = DataSet.get_params_mask(config)
        collate_fn = DataSet.wrapper_mask(DataSet.collate_fn_many_args, mc_params = True, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
        model = self.model
        model.eval()
        latent_list = []
        params = []
        loss =  []
        
        with torch.no_grad():
            for x,par in tqdm(test_loader):
                x = x.to(device)
                mu, log_var, (h_n, c_n) = model.encoder(x)
                recon_x, mu, log_var, pred_num = self.model(x)
                recon_loss, kl_divergence, num_det_loss = Loss.vae_loss(recon_x, x, mu, log_var, pred_num, mask=self.mask,
                                                                        use_mask=self.use_mask, koef_loss=self.koef_loss, reduction='none')
                latent_list.append(mu.cpu())
                params.append(par.cpu())
                loss.append(recon_loss.cpu())
        latent_list = torch.cat(latent_list, dim=0)
        params = torch.cat(params, dim=0)
        loss = torch.cat(loss, dim=0)
        print(params.shape)
        try:
            writer.add_embedding(latent_list,
                        metadata=params[:,1],
                        )
        except Exception as e:
            # logger.log_exception(e)
            print(e)
        return latent_list, params, loss

if __name__ == "__main__":
    config = 'config.yaml'
    pipline = Pipline(config)
    pipline.train()