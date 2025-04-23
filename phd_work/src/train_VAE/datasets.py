import torch
from typing import Optional, Tuple, Union, List
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import h5py as h5
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
class VariableLengthDataset(Dataset):
    def __init__(self, data_path:str, mode:str, mc_params:bool = False, paticles: Optional[List[str]] = None):
        """
        Args:
            data: список тензоров, где каждый тензор имеет форму (seq_len, 6)
        """
        
        self.mass_dict = {'pr': 14,
                     'photon': 1,
                     'fe': 5626}
        self.paticles = paticles
        print('mc_params',mc_params)
        data, ev_starts, mc_params = self.read_h5(data_path, mode, mc_params, paticles)
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
        
        mc_params = self.mc_params[idx]
        if self.mc_params is not None:
            return torch.tensor(self.data[st:fn]), torch.tensor(mc_params[1]), torch.tensor(mc_params)
        else:
            return torch.tensor(self.data[st:fn]), torch.tensor(mc_params[1])
    def read_h5(self, data_path, mode, mc_params, paticles: Optional[List[str]] = None):
        with h5.File(data_path,'r') as f:
            print('keys', list(f.keys()))
            train = f[mode]
            dt_params = torch.tensor(train['dt_params'][()])
            ev_starts = torch.tensor(train['ev_starts'][()])
            if mc_params:
                mc_params = torch.tensor(train['mc_params'][()])
                if paticles is not None:
                    dt_params, ev_starts, mc_params = self.choise_def_particles(paticles, data = dt_params, ev_starts=ev_starts, mc_params=mc_params, get_mc_params=True)
            else:
                mc_params = torch.tensor(train['mc_params'][()])
                if paticles is not None:
                    dt_params, ev_starts = self.choise_def_particles(paticles, data = dt_params, ev_starts=ev_starts, mc_params=mc_params, get_mc_params=False)
            # mc_params = None
        return dt_params, ev_starts, mc_params
    def str2mass(self, name: List[str]) -> List[int]:
        #1. mc_parttype (CORSIKA, 1 - gamma, 14 - proton, 5626 - Fe)
        mass_dict = self.mass_dict
        return [mass_dict[n] for n in name]
    def choise_def_particles_2(self, name: List[str],data, ev_starts, mc_params, par_num: int = 1, get_mc_params: bool = False):
        mass = self.str2mass(name)
        data_shape = list(data.shape)
        data_shape[0]=0
        data_shape=tuple(data_shape)
        data_new = torch.zeros(data_shape, dtype=data.dtype, device=data.device)
        ev_starts_new = torch.tensor([0], dtype=torch.long)
        if get_mc_params:
            mc_params_shape = list(mc_params.shape)
            mc_params_shape[0]=0
            mc_params_shape=tuple(mc_params_shape)
            mc_params_new = torch.zeros(mc_params_shape, dtype=mc_params.dtype, device=mc_params.device)
        for i in tqdm(range(len(mc_params))):
            p=mc_params[i,par_num]
            if p in mass:
                ev_s = ev_starts[i]
                ev_f = ev_starts[i+1]
                data_new = torch.concat([data_new, data[ev_s:ev_f]], dim=0)
                ev_starts_new = torch.concat([ev_starts_new, torch.tensor([ev_f-ev_s], dtype=torch.long)])
                if get_mc_params:
                    mc_params_new = torch.concat([mc_params_new, mc_params[i]], dim=1)
        if get_mc_params:
            return data_new, ev_starts_new, mc_params_new
        else:
            return data_new, ev_starts_new
    def choise_def_particles(self, name: List[str], data, ev_starts, mc_params, par_num: int = 1, get_mc_params: bool = False):
        # Преобразуем имена частиц в массы
        mass = self.str2mass(name)
        
        # Создаем маску для выбора нужных частиц
        mask = torch.isin(mc_params[:, par_num], torch.tensor(mass, device=mc_params.device))
        
        # Применяем маску к mc_params и ev_starts
        mc_params_filtered = mc_params[mask]
        ev_starts_filtered = ev_starts[:-1][mask]  # Исключаем последний элемент ev_starts, так как он не соответствует событию
        
        # Вычисляем новые индексы начала событий
        ev_starts_new = torch.cat([torch.tensor([0], device=data.device), torch.cumsum(ev_starts[1:][mask] - ev_starts[:-1][mask], dim=0)])
        
        # Собираем данные, соответствующие выбранным событиям
        try:
            data_indices = torch.cat([torch.arange(ev_starts[i], ev_starts[i+1], device=data.device) for i in torch.where(mask)[0]])
            data_new = data[data_indices]
        except NotImplementedError:
            raise NotImplementedError(f"Probably this particle was not find in this file. Total ev_starts is {len(ev_starts)}")
        if get_mc_params:
            return data_new, ev_starts_new, mc_params_filtered
        else:
            return data_new, ev_starts_new
    def cut_ev_start(self, name: List[str], ev_starts, mc_params, par_num: int = 1, get_mc_params: bool = False):
        mass = self.str2mass(name)
        data = torch.tensor([], dtype=torch.long)  # Используем CPU
        for i, m in enumerate(mass):
            where_ = torch.where(mc_params[:, par_num] == m)[0]
            where = torch.cat((where, where_))
            del where_  # Освобождаем память
            torch.cuda.empty_cache()  # Очищаем кэш CUDA
        ev_starts = ev_starts[where]
        mc_params = mc_params[where]
        if get_mc_params:
            return ev_starts, mc_params
        else:
            return ev_starts
    
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
        particles = None
        for item, part, par  in batch:
            sequences.append(torch.cat((start_token, item, stop_token), dim=0))
            if params is None:
                params = par.unsqueeze(0)
                particles = part.unsqueeze(0)
            else:
                params = np.concatenate((params, par.unsqueeze(0)), axis=0)
                particles = np.concatenate((particles, part.unsqueeze(0)), axis=0) 
        # Заполняем последовательности до одинаковой длины
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value)  # (batch_size, max_seq_len, 5)
        return padded_sequences, torch.tensor(particles), torch.tensor(params)
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

    def wrapper_fNoneunc(batch):
        return func(batch, **kwargs)

    return wrapper_func
