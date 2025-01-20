import torch
from typing import Optional, Tuple, Union
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import h5py as h5
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class VariableLengthDataset(Dataset):
    def __init__(self, data_path:str, mode:str, mc_params:bool = False):
        """
        Args:
            data: список тензоров, где каждый тензор имеет форму (seq_len, 6)
        """
        data, ev_starts, mc_params = self.read_h5(data_path, mode, mc_params)
        # prepoccessing
        # data = self.preprocc_signal(data, 3)
        self.data = data
        self.ev_starts = ev_starts
        self.mc_params = mc_params
    def __len__(self):
        return len(self.ev_starts)-1
    def preprocc_signal(self, data: np.ndarray, ch: int = 3) -> np.ndarray:
        data[:,ch] = np.log(data[:,ch] - data[:,ch].min()+1e-6)
        return data
    def __getitem__(self, idx):
        st = self.ev_starts[idx]
        fn = self.ev_starts[idx + 1]
        if self.mc_params is not None:
            mc_params = self.mc_params[idx]
            return torch.tensor(self.data[st:fn]), torch.tensor(mc_params)
        else:
            return torch.tensor(self.data[st:fn])
    def read_h5(self, data_path, mode, mc_params):
        with h5.File(data_path,'r') as f:
            print('keys', list(f.keys()))
            train = f[mode]
            dt_params = train['dt_params'][()]
            ev_starts = train['ev_starts'][()]
            if mc_params:
                mc_params = train['mc_params'][()]
            else: 
                mc_params = None
        return dt_params, ev_starts, mc_params
    
    
def get_params_mask(config):
    """
    This function extracts and prepares start, stop tokens, and padding value from a given configuration dictionary.

    Parameters:
    - config (dict): A dictionary containing the configuration parameters. It should have the following keys:
        - 'start_token': The value to be used as the start token.
        - 'stop_token': The value to be used as the stop token.
        - 'padding_value': The value to be used for padding sequences.

    Returns:
    - dict: A dictionary containing the prepared start token, stop token, and padding value. The dictionary has the following keys:
        - 'start_token': A tensor of shape (1,6) filled with the start token value.
        - 'stop_token': A tensor of shape (1,6) filled with the stop token value.
        - 'padding_value': The padding value.
    """
    start_token = config['start_token']
    stop_token = config['stop_token']
    padding_value = config['padding_value']

    if isinstance(start_token,int):
        start_token = torch.ones((1,6)) * start_token
    elif isinstance(start_token, str):
    
        if start_token == 'hard_v1':
            # TODO: change  
            # signal min -0.277908
            # flat min -8.798042
            # real-flat max 15.595449
            start_token = torch.tensor([0, 0, 0, -0.277908, -8.798042, 15.595449]).unsqueeze(0)
        else:
            raise ValueError(f"Unknown start_token: {start_token}")
    else:
        raise ValueError(f"Unknown start_token: {start_token}")

    stop_token = torch.ones((1,6)) * stop_token
    return {'start_token': start_token, 'stop_token': stop_token, 'padding_value': padding_value}

# @wrapper_func()
def collate_fn_many_args(batch: Tensor, start_token: Tensor, stop_token: Tensor, padding_value: int, mc_params: bool = False) -> Tensor:
    #, padding_value, star_token, stop_toke
    """
    Кастомная функция для DataLoader, которая заполняет последовательности до максимальной длины в батче.

    Предается много аргументов. Поэтому преед использованием в DataLoader надо сделать wrapper_mask(collate_fn_many_args, ...)
    """
    # start_token, stop_token = torch.zeros((1,6)), torch.zeros((1,6))
    # Извлекаем каждую последовательность в батче
    if mc_params:
        sequences = []
        params = None
        for item, par in batch:
            sequences.append(torch.cat((start_token, item, stop_token), dim=0))
            if params is None:
                params = par.unsqueeze(0)
            else:
                params = np.concatenate((params, par.unsqueeze(0)), axis=0)
        # Заполняем последовательности до одинаковой длины
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value)  # (batch_size, max_seq_len, 5)
        return padded_sequences, torch.tensor(params)
    else: 
        sequences = [torch.cat((start_token, item, stop_token), dim=0) for item in batch]
        # Заполняем последовательности до одинаковой длины
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value)  # (batch_size, max_seq_len, 5)
        return padded_sequences
def wrapper_mask(func, *args, **kwargs):
    """
    A wrapper function that adds start and stop tokens to each sequence in a batch and pads them to the same length.

    Parameters:
    - func (function): The original function to be wrapped. It should accept a batch of sequences, start token, stop token, and padding value as parameters.
    - *args (tuple): Additional positional arguments to be passed to the original function.
    - **kwargs (dict): Additional keyword arguments to be passed to the original function. It should include 'padding_value', 'start_token', and 'stop_token'.

    Returns:
    - wrapper_func (function): The wrapped function that applies the start and stop tokens, pads the sequences, and calls the original function.
    """
    padding_value = kwargs['padding_value']
    start_token = kwargs['start_token']
    stop_token = kwargs['stop_token']

    def wrapper_func(batch):
        return func(batch, **kwargs)

    return wrapper_func
