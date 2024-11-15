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
print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
data_path = '/home/rfit/Telescope_Array/phd_work/data/normed/pr_q4_14yr_e1_0110_excl_sat_F_excl_geo_F.h5'

import model as Model
import datasets as DataSet
import loss as Loss

from torch.utils.tensorboard import SummaryWriter
import yaml

def read_config(config: str = 'config.yaml'):
    with open(config, 'r') as file:
        hparams = yaml.safe_load(file)
    return hparams
def prepipline(config):

    name = config['PATH'].split('/')[-1]
    writer = SummaryWriter(log_dir=os.path.join('runs', name))
    writer.add_text('hparams',  str(config))

    dataset = DataSet.VariableLengthDataset(data_path, 'train')
    val_dataset = DataSet.VariableLengthDataset(data_path, 'test')

    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=DataSet.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=DataSet.collate_fn)
    model = Model.VAE(config['input_dim'], config['hidden_dim'], config['latent_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config['lr']))
    return {'train_loader': train_loader, 'val_loader': val_loader, 'model': model, 'optimizer': optimizer, 'writer': writer}
# Цикл обучения (предполагается, что train_loader предоставляет пакеты последовательностей переменной длины)
def train(config):
    prepipline_dict = prepipline(config)
    train_loader = prepipline_dict['train_loader']
    val_loader = prepipline_dict['val_loader']
    model = prepipline_dict['model']
    optimizer = prepipline_dict['optimizer']
    writer = prepipline_dict['writer'] 
    PATH = config['PATH']
    epochs = config['epoches']
    os.makedirs(PATH, exist_ok = True)
    iters = 0
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc =f"TRAIN Epoch {epoch + 1}/{epochs}, Loss: 0.0")
        for x in pbar:  # x должен быть пакетом последовательностей с заполнением
            x = x.to(device)
            optimizer.zero_grad()
            recon_x, mu, log_var = model(x)
            recon_loss, kl_divergence = Loss.vae_loss(recon_x, x, mu, log_var)
            loss = recon_loss + kl_divergence
            writer.add_scalar("train/Loss", loss, iters)
            writer.add_scalar("train/KL_loss", kl_divergence, iters)
            writer.add_scalar("train/recon_loss", recon_loss, iters)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"TRAIN Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
            iters += 1
        model.eval()
        loss_mean = []
        KL_loss_mean = []
        recon_loss_mean = []
        
        pbar_val = tqdm(val_loader, desc =f"VAL Epoch {epoch + 1}/{epochs}, Loss: 0.0")
        for x in pbar_val:  # x должен быть пакетом последовательностей с заполнением
            x = x.to(device)
            recon_x, mu, log_var = model(x)
            recon_loss, kl_divergence = Loss.vae_loss(recon_x, x, mu, log_var)
            loss = recon_loss + kl_divergence
            loss_mean.append(loss.item())
            KL_loss_mean.append(loss.item())
            recon_loss_mean.append(kl_divergence.item())
            pbar_val.set_description(f"VAL Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), os.path.join(PATH, f'epoch_{epoch}'))
        print(f'Epoch {epoch + 1}, Loss: {np.array(loss_mean).mean()}')
        writer.add_scalar("val/Loss", np.array(loss_mean).mean(), epoch)
        writer.add_scalar("val/KL_loss", np.array(KL_loss_mean).mean(), epoch)
        writer.add_scalar("val/recon_loss", np.array(recon_loss_mean).mean(), epoch)

if __name__ == '__main__':
    config = read_config()
    train(config)