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
    print(os.listdir())
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


# Tresh
    # Unneded
# def choise_def_particles_2(self, name: List[str],data, ev_starts, mc_params, par_num: int = 1, get_mc_params: bool = False):
#     mass = self.str2mass(name)
#     data_shape = list(data.shape)
#     data_shape[0]=0
#     data_shape=tuple(data_shape)
#     data_new = torch.zeros(data_shape, dtype=data.dtype, device=data.device)
#     ev_starts_new = torch.tensor([0], dtype=torch.long)
#     if get_mc_params:
#         mc_params_shape = list(mc_params.shape)
#         mc_params_shape[0]=0
#         mc_params_shape=tuple(mc_params_shape)
#         mc_params_new = torch.zeros(mc_params_shape, dtype=mc_params.dtype, device=mc_params.device)
#     for i in tqdm(range(len(mc_params))):
#         p=mc_params[i,par_num]
#         if p in mass:
#             ev_s = ev_starts[i]
#             ev_f = ev_starts[i+1]
#             data_new = torch.concat([data_new, data[ev_s:ev_f]], dim=0)
#             ev_starts_new = torch.concat([ev_starts_new, torch.tensor([ev_f-ev_s], dtype=torch.long)])
#             if get_mc_params:
#                 mc_params_new = torch.concat([mc_params_new, mc_params[i]], dim=1)
#     if get_mc_params:
#         return data_new, ev_starts_new, mc_params_new
#     else:
#         return data_new, ev_starts_new