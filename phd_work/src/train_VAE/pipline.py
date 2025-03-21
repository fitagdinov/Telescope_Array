from tqdm import tqdm
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
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

logger = logging.getLogger()
class Pipline():
    def __init__(self, config, need_train_DS: bool = True, many_val_loaders: bool = True):
        config = read_config(config)
        self.config = config
        self.prepipline_dict = self.prepipline(config, need_train_DS=need_train_DS, many_val_loaders= many_val_loaders)
        self.pretrain()
    def prepipline(self, config, need_train_DS: bool = True, many_val_loaders: bool = True):
        name = config['PATH'].split('/')[-1]
        writer = SummaryWriter(log_dir=os.path.join('/home/rfit/Telescope_Array/phd_work/TBruns/', config['exp'], name))
        writer.add_text('hparams',  str(config))
        kwargs = DataSet.get_params_mask(config)
        kwargs['mc_params'] = True
        collate_fn = DataSet.wrapper_mask(DataSet.collate_fn_many_args, **kwargs)
        if need_train_DS:
            dataset = DataSet.VariableLengthDataset(config['data_path'], 'train', paticles=config['paticles']['train'], mc_params=True)
            train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
        else:
            dataset = None
            train_loader = None
        # for many particles
        if many_val_loaders:
            val_loaders = []
            for p in config['paticles']['test']:
                val_dataset = DataSet.VariableLengthDataset(config['data_path'], 'test', paticles=[p], mc_params=True)
                val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
                val_loaders.append(val_loader)
        else:
            val_dataset = DataSet.VariableLengthDataset(config['data_path'], 'test', paticles=config['paticles'], mc_params=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
            val_loaders = [val_loader]        
        start_token = kwargs['start_token'].to(device)
        model = Model.VAE(config['input_dim'], config['hidden_dim'], config['latent_dim'], lstm2=config['lstm2'], start_token=start_token).to(device)
        if config['chpt'] != 'None':
            model.load(config['chpt'])
        optimizer = optim.Adam(model.parameters(), lr=float(config['lr']))
        # write augmentes
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.model = model
        self.writer = writer
        return {'train_loader': train_loader, 'val_loaders': val_loaders, 'model': model, 'optimizer': optimizer, 'writer': writer}
    def load_chpt(self, chpt_path: str):
        self.model.load(chpt_path)
    def pretrain(self):
        config = self.config
        PATH = os.path.join(config['save_model_path'], config['PATH'])
        self.PATH = PATH
        config['PATH'] = PATH
        print("Saving Path: {}".format(PATH))
        prepipline_dict = self.prepipline_dict
        self.train_loader = prepipline_dict['train_loader']
        self.val_loaders = prepipline_dict['val_loaders']
        self.model = prepipline_dict['model']
        self.optimizer = prepipline_dict['optimizer']
        self.writer = prepipline_dict['writer']
        self.epochs = config['epoches']
        self.mask = config['padding_value']
        self.show_index = config['show_index']
        self.koef_KL = config['koef_KL']
        self.koef_DL = config['koef_DL']
        self.koef_mass = config['koef_mass']

        self.use_mask = config['use_mask']
        self.stop_token = config['stop_token']
        self.start_token = config['start_token']
        koef_loss = torch.tensor(config['koef_loss']).unsqueeze(0).to('cpu')
        self.paticles = self.config['paticles']
        self.koef_loss = koef_loss.to(device)
        os.makedirs(PATH, exist_ok = True)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.2, patience=5, threshold=0.005,)
    def train(self):
        self.loss_best = 1000
        iters = 0
        for epoch in range(self.epochs):
            print('lr_scheduler', self.optimizer.param_groups[0]['lr'])
            self.writer.add_scalar("lr_scheduler", self.optimizer.param_groups[0]['lr'], epoch)
            self.model.train()
            pbar = tqdm(self.train_loader, desc =f"TRAIN Epoch {epoch + 1}/{self.epochs}, Loss: 0.0")
            for x, part, _ in pbar:  # x должен быть пакетом последовательностей с заполнением
                x = x.to(device)
                part = torch.where(part == 1, 0, 1).to(device) # 0- photon, 1- proton
                self.optimizer.zero_grad()
                recon_x, mu, log_var, pred_num, pred_mass = self.model(x)
                recon_loss, kl_divergence, num_det_loss, mass_loss = Loss.vae_loss(recon_x, x, mu, log_var, pred_num, pred_mass, part,
                                                                                    mask=self.mask, use_mask=self.use_mask, koef_loss=self.koef_loss
                                                                                    )
                num_det_loss *= self.koef_DL
                kl_divergence *= self.koef_KL
                mass_loss *= self.koef_mass
                loss = recon_loss + kl_divergence + num_det_loss + mass_loss
                self.writer.add_scalar("train/Loss", loss, iters)
                self.writer.add_scalar("train/KL_loss", kl_divergence, iters)
                self.writer.add_scalar("train/recon_loss", recon_loss, iters)
                self.writer.add_scalar("train/num_det_loss", num_det_loss, iters)
                self.writer.add_scalar("train/mass_loss", mass_loss, iters)
                loss.backward()
                self.optimizer.step()
                pbar.set_description(f"TRAIN Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")
                iters += 1
            
            self.validation(epoch=epoch, analys=True)
    def validation(self, epoch: Optional[int] = None, analys:bool=False) -> None:
        """
        Performs validation on the model using the validation dataset.

        Parameters:
        epoch (int): The current epoch number.
        analys (bool): A flag indicating whether to perform analysis during validation.

        Returns:
        None. The function updates the model's state_dict if the validation loss is the best so far.
        It also prints the validation loss and loss_best, updates the learning rate scheduler,
        and writes the validation metrics to TensorBoard.
        """
        epoch = epoch if epoch is not None else -1
        model = self.model
        val_loaders = self.val_loaders
        epochs = self.epochs
        koef_KL = self.koef_KL
        koef_DL = self.koef_DL

        model.eval()
        loss_mean = np.array([])
        KL_loss_mean = np.array([])
        recon_loss_mean = np.array([])
        num_det_loss_mean = np.array([])
        for i, val_loader in enumerate(val_loaders):
            particle = self.config['paticles']['test'][i]
            loss, KL, recon, num_det = self.validation_step(epoch=epoch, val_loader=val_loader, model=model, koef_KL=koef_KL, koef_DL=koef_DL,particle=particle)
            loss_mean = np.concatenate((loss_mean, loss))
            KL_loss_mean = np.concatenate((KL_loss_mean, KL))
            recon_loss_mean = np.concatenate((recon_loss_mean, recon))
            num_det_loss_mean = np.concatenate((num_det_loss_mean, num_det))
            loss_final = loss_mean.mean()
        # write in TB
        self.writer.add_scalar("val/Loss/all", loss_final, epoch)
        # self.writer.add_scalar("val/KL_loss/all", KL_loss_mean.mean(), epoch)
        self.writer.add_scalar("val/recon_loss/all", recon_loss_mean.mean(), epoch)
        self.writer.add_scalar("val/num_det_loss/all", num_det_loss_mean.mean(), epoch)
        if analys:
            if epoch>0:
                self.scheduler.step(loss_final)
            if loss_final<self.loss_best:
                self.loss_best = loss_final
                torch.save(model.state_dict(), os.path.join(self.PATH, f'best'))

            print(f'Epoch {epoch + 1}, Loss: {loss_final} loss_best {self.loss_best}')

    def validation_step(self, epoch, val_loader, model, koef_KL=1, koef_DL=1, particle = None):
        particle = '' if particle is None else particle
        loss_mean = []
        KL_loss_mean = []
        recon_loss_mean = []
        num_det_loss_mean = []

        pred_mass_array = None
        real_mass_array = None
        pbar_val = tqdm(val_loader, desc =f"VAL Epoch {epoch + 1} in {particle}, Loss: 0.0")
        for x, part, _ in pbar_val:  # x should be a batch of sequences with padding
            x = x.to(device)
            part = torch.where(part == 1, 0, 1).to(device) # 0- photon, 1- proton
            recon_x, mu, log_var, pred_num, pred_mass = model(x)
            recon_loss, kl_divergence, num_det_loss, mass_loss = Loss.vae_loss(recon_x, x, mu, log_var, pred_num, pred_mass, part,
                                                                                    mask=self.mask, use_mask=self.use_mask, koef_loss=self.koef_loss
                                                                                    )
            mass_loss *= self.koef_mass
            kl_divergence *= koef_KL
            num_det_loss *= koef_DL
            loss = recon_loss + kl_divergence + num_det_loss
            loss_mean.append(loss.item())
            KL_loss_mean.append(kl_divergence.item())
            recon_loss_mean.append(recon_loss.item())
            num_det_loss_mean.append(num_det_loss.item())
            pbar_val.set_description(f"VAL Epoch {epoch + 1} in {particle}, Loss: {loss.item():.4f}")

        loss_final = np.array(loss_mean).mean()
        # write in TB
        self.writer.add_scalar(f"val/Loss/{particle}", loss_final, epoch)
        # self.writer.add_scalar(f"val/KL_loss/{particle}", np.array(KL_loss_mean).mean(), epoch)
        self.writer.add_scalar(f"val/recon_loss/{particle}", np.array(recon_loss_mean).mean(), epoch)
        self.writer.add_scalar(f"val/num_det_loss/{particle}", np.array(num_det_loss_mean).mean(), epoch)

        #show from last batch
        real = x[self.show_index]
        fake = recon_x[self.show_index]
        num = pred_num[self.show_index]
        for ii in range(len(self.show_index)):
            # get from back side
            i = -ii
            fig = show_pred(real[i], fake[i], tokens = (self.start_token, self.stop_token, self.mask), lenght_predict = num[i])
            self.writer.add_figure(f"val/show_pred_{ii}/{particle}", fig, epoch)
        return np.array(loss_mean), np.array(KL_loss_mean), np.array(recon_loss_mean), np.array(num_det_loss_mean)

    def predict_latent(self, write_embedding: bool = True, choise_num:Optional[int] = None):
        # TODO сделать в отдельной функции getl loader
        config = self.config
        writer = SummaryWriter(log_dir=os.path.join('runs_tests', 'tests'))
        model = self.model
        model.eval()
        latent_list = []
        params = []
        particles = []
        all_loss = None
        test_loaders = self.val_loaders
        dict_info = {}
        with torch.no_grad():
            for i, test_loader in enumerate(test_loaders):
                for x in tqdm(test_loader):
                    x = x.to(device)
                    mu, log_var, (h_n, c_n) = model.encoder(x)
                    recon_x, mu, log_var, pred_num = model(x)
                    recon_loss, kl_divergence, num_det_loss = Loss.vae_loss(recon_x, x, mu, log_var, pred_num, mask=self.mask, use_mask = self.config['use_mask'], koef_loss=self.koef_loss )
                    print(recon_loss)
                    if all_loss is None:
                        all_loss = recon_loss.cpu().detach() 
                    else:
                        print(all_loss.shape)
                        all_loss = torch.cat((all_loss, recon_loss.cpu().detach()))
                    latent_list.append(mu.cpu())
                    # params.append(par.cpu())
                    particles += [self.config['paticles'][i]] * mu.shape[0] # for equal lenght with latent

                    #write in dict
                    # TODO otimize
                    try:
                        dict_info[self.config['paticles'][i]] = torch.cat((dict_info[self.config['paticles'][i]], mu.cpu().detach() ), dim=0)
                    except KeyError:
                        dict_info[self.config['paticles'][i]] = mu.cpu().detach() 
        latent_list = torch.cat(latent_list, dim=0)
        # params = torch.cat(params, dim=0)
        # print(params.shape)
        if choise_num:
            latent_list, particles, all_loss = self.select_random_ordered(latent_list, particles, all_loss, int(choise_num))
        print(latent_list. shape, len(particles))
        try:
            if write_embedding:
                writer.add_embedding(latent_list,
                            metadata=particles,
                            )
        except Exception as e:
            # logger.log_exception(e)
            print(e)
        for p,lat in dict_info.items():
            print(f"particles:{p} mean{lat.mean()} std {lat.std()}")
            # for i in range(lat.shape[1]):
            #     print(f"in {i} comonent mean{lat[:,i].mean()} std {lat[:,i].std()}")
        return latent_list, particles, all_loss#params
    def select_random_ordered(self, latent_list, particles, all_loss, m):
        n = len(particles)
        assert n == latent_list.shape[0], "Размеры данных не совпадают"
        assert m <= n, "m не может превышать n"

        # Генерация случайных индексов без повторений
        indices = np.random.choice(n, size=m, replace=False)
        
        # Сортировка индексов для сохранения порядка
        sorted_indices = np.sort(indices)
        
        # Выборка данных
        selected_latent = latent_list[sorted_indices]
        selected_particles = [particles[i] for i in sorted_indices]
        all_loss = all_loss[sorted_indices]
        return selected_latent, selected_particles, all_loss

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="A simple example of argparse")

    # Add optional arguments
    parser.add_argument("-m", "--mode", type=str, help="The output file to save results", default="train")
    parser.add_argument("-e", "--write_embading", type=bool, help="Write latent data in TB for project analys", default="True")
    # Parse the arguments
    args = parser.parse_args()

    config = 'config.yaml'
    if args.mode == 'train':
        pipline = Pipline(config)
        print('TRAIN PIPLINE')
        pipline.train()
    elif args.mode == 'test':
        pipline = Pipline(config, need_train_DS=False)
        print('TEST PIPLINE')
        pipline.validation()
    elif args.mode == 'latent':
        pipline = Pipline(config, need_train_DS=False)
        print('Latent PIPLINE')
        pipline.predict_latent(args.write_embading)