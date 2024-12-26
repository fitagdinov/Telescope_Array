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
from typing import Optional, Tuple, Union
import pytorch_warmup as warmup
from  torch.optim.lr_scheduler import ExponentialLR

from torch.utils.tensorboard import SummaryWriter
import yaml
import time
def get_time() -> str:
    """
    This function retrieves the current time and formats it into a string.

    Parameters:
    None

    Returns:
    str: The current time in the format 'dd.mm.yyyy_HH:MM'.
    """
    sec = time.time()
    struct = time.localtime(sec)
    return time.strftime('%d.%m.%Y %H:%M', struct).replace(' ','_')

def read_config(config: str = 'config.yaml') -> dict:
    """
    This function reads a configuration file in YAML format and returns its contents as a dictionary.

    Parameters:
    - config (str): The path to the configuration file. The default value is 'config.yaml'.

    Returns:
    dict: A dictionary containing the configuration parameters.
    """
    with open(config, 'r') as file:
        hparams = yaml.safe_load(file)
    return hparams
def get_params_str(config: dict) -> str:
    """
    This function generates a string representation of specific parameters from a given configuration dictionary.

    Parameters:
    - config (dict): A dictionary containing configuration parameters. It should contain a list of parameter names under the key 'write_param'.

    Returns:
    str: A string representation of the specified parameters in the format 'param1=value1;param2=value2;...'.
    """
    write_param = config['write_param']
    res = ""
    for p in write_param:
        res += f'{p}={config[p]};_'
    return res
def clean_mask(data: torch.Tensor, tokens: Optional[Tuple[int, int, int]] =None, lenght: Optional[int]=None) -> torch.Tensor:
    """
    This function cleans a masked tensor by removing start, end, and mask tokens, and adjusting the length accordingly.

    Parameters:
    - data (torch.Tensor): The input tensor to be cleaned. It should be of shape (sequence_length, features).
    - tokens (Optional[Tuple[int, int, int]]): A tuple containing the start token, end token, and mask token. If None, default values will be used.

    Returns:
    torch.Tensor: The cleaned tensor with the start, end, and mask tokens removed, and the length adjusted accordingly.
    """
    if lenght is not None:
        return data[:lenght], lenght
    st, fn, ms = tokens
    mask = torch.where(data != ms, 1, 0)
    lenght = torch.sum(mask[:,0]) # length but used st and fn tokens
    if fn is not None:
        lenght-=1 # fn token is not real data
    if st is not None:
        data = data[1:] # del first data
        lenght-=1
    return data[:lenght], lenght   
    
def show_pred(data, fake, tokens: Optional[Tuple[int, int, int]]=None,
               lenght_predict: Union[np.ndarray, torch.Tensor] = None) -> plt.figure:
    '''
    data - shape (det, featches)

    fake - shape (det, featches)

    return: fig
    '''

    #TODO переписать нормально
    data, real_lenght = clean_mask(data, tokens = tokens)
    if lenght_predict is not None:
        fake_lenght = float(lenght_predict)
        if tokens[0] is not None:
            # have start token
            fake = fake[1:]
        fake, _ = clean_mask(fake, tokens = tokens, lenght=real_lenght)
    else:
        if tokens[0] is not None:
            # have start token
            fake = fake[1:]
        fake, fake_lenght = clean_mask(fake, tokens = tokens, lenght=real_lenght)
    names = ['det x', 'det y', 'det z', 'signal', 'flat front', '(real - front)']
    fig, axs = plt.subplots(2,3, figsize = (10,10))
    for i in range(6):
        row = i%2
        col = i//2
        axs[row][col].plot(fake.to('cpu').detach().numpy()[:,i], 'r')
        axs[row][col].plot(data.to('cpu').detach().numpy()[:,i], 'b')
        axs[row][col].legend(['fake', 'true'])
        axs[row][col].set_title(f'chanal {names[i]}')
        axs
    lenght_info = f'real_lenght {real_lenght}\nfake_lenght {fake_lenght}'
    plt.suptitle(lenght_info)
    return fig

def prepipline(config):
    name = config['PATH'].split('/')[-1]
    writer = SummaryWriter(log_dir=os.path.join('runs', name))
    writer.add_text('hparams',  str(config))

    dataset = DataSet.VariableLengthDataset(data_path, 'train')
    val_dataset = DataSet.VariableLengthDataset(data_path, 'test')
    kwargs = DataSet.get_params_mask(config)
    collate_fn = DataSet.wrapper_mask(DataSet.collate_fn_many_args, **kwargs)
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    start_token = kwargs['start_token'].to(device)
    model = Model.VAE(config['input_dim'], config['hidden_dim'], config['latent_dim'], start_token=start_token).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config['lr']))
    return {'train_loader': train_loader, 'val_loader': val_loader, 'model': model, 'optimizer': optimizer, 'writer': writer}
# Цикл обучения (предполагается, что train_loader предоставляет пакеты последовательностей переменной длины)
def train(config):
    # add info in save path
    PATH = config['PATH'] + get_time() + get_params_str(config)
    config['PATH'] = PATH
    print("Saving Path: {}".format(PATH))
    prepipline_dict = prepipline(config)
    train_loader = prepipline_dict['train_loader']
    val_loader = prepipline_dict['val_loader']
    model = prepipline_dict['model']
    optimizer = prepipline_dict['optimizer']
    writer = prepipline_dict['writer']
    epochs = config['epoches']
    mask = config['padding_value']
    show_index = config['show_index']
    koef_KL = config['koef_KL']
    koef_DL = config['koef_DL']

    use_mask = config['use_mask']
    stop_token = config['stop_token']
    start_token = config['start_token']
    os.makedirs(PATH, exist_ok = True)
    iters = 0

    # warmup NEED PYTHON >=3.9
    # num_steps = len(train_loader) * epochs
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    # scheduler = ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc =f"TRAIN Epoch {epoch + 1}/{epochs}, Loss: 0.0")
        for x in pbar:  # x должен быть пакетом последовательностей с заполнением
            x = x.to(device)
            optimizer.zero_grad()
            recon_x, mu, log_var, pred_num = model(x)
            recon_loss, kl_divergence, num_det_loss = Loss.vae_loss(recon_x, x, mu, log_var, pred_num, mask=mask, use_mask=use_mask)
            num_det_loss *= koef_DL
            kl_divergence *= koef_KL
            loss = recon_loss + kl_divergence + num_det_loss
            writer.add_scalar("train/Loss", loss, iters)
            writer.add_scalar("train/KL_loss", kl_divergence, iters)
            writer.add_scalar("train/recon_loss", recon_loss, iters)
            writer.add_scalar("train/num_det_loss", num_det_loss, iters)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"TRAIN Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
            iters += 1
        # writer.add_scalar("LR",[i['lr'] for i in optim.param_groups][0], epoch)
        # scheduler.step()
            # with warmup_scheduler.dampening():
            #     lr_scheduler.step()
        model.eval()
        loss_mean = []
        KL_loss_mean = []
        recon_loss_mean = []
        num_det_loss_mean = []
        pbar_val = tqdm(val_loader, desc =f"VAL Epoch {epoch + 1}/{epochs}, Loss: 0.0")
        for x in pbar_val:  # x должен быть пакетом последовательностей с заполнением
            x = x.to(device)
            recon_x, mu, log_var, pred_num = model(x)
            recon_loss, kl_divergence, num_det_loss = Loss.vae_loss(recon_x, x, mu, log_var, pred_num, mask=mask, use_mask = config['use_mask'])
            kl_divergence *= koef_KL
            num_det_loss *= koef_DL
            loss = recon_loss + kl_divergence + num_det_loss
            loss_mean.append(loss.item())
            KL_loss_mean.append(kl_divergence.item())
            recon_loss_mean.append(recon_loss.item())
            num_det_loss_mean.append(num_det_loss.item())
            pbar_val.set_description(f"VAL Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), os.path.join(PATH, f'epoch_{epoch}'))
        print(f'Epoch {epoch + 1}, Loss: {np.array(loss_mean).mean()}')
        writer.add_scalar("val/Loss", np.array(loss_mean).mean(), epoch)
        writer.add_scalar("val/KL_loss", np.array(KL_loss_mean).mean(), epoch)
        writer.add_scalar("val/recon_loss", np.array(recon_loss_mean).mean(), epoch)
        writer.add_scalar("val/num_det_loss", np.array(num_det_loss_mean).mean(), epoch)

        #show from last batch
        real = x[show_index]
        fake = recon_x[show_index]
        num = pred_num[show_index]
        for ii in range(len(show_index)):
            # get from back side
            i = -ii
            fig = show_pred(real[i], fake[i], tokens = (start_token, stop_token, mask), lenght_predict = num[i])
            writer.add_figure(f"val/show_pred_{i}", fig, epoch)            

if __name__ == '__main__':
    config = read_config()
    train(config)