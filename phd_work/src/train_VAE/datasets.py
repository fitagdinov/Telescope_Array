from functools import partial
import torch
from typing import Optional, Tuple, Union, List
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import h5py as h5
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm



class VariableLengthDataset(Dataset):
    """
    dt_bunlde (num_dets,6,6,7):
    parameters of detectors bundle
    0. detector x relative to shower core, 1200 m units
    1. detector y relative to shower core, 1200 m units
    2. detector z, 1200 m units
    3. detector signal MIP
    4. time of the plane front arrival, mks
    5. time of the waveform relative to the plane front, mks
    6. mask for trigered detectors

    recos (num_evs,15):
    0. theta
    1. phi
    2. S_800
    3. E_gamma
    4. d_border, km
    5. chi2/ndof for joint fit
    6. Linsley front curvature
    7. Area-over-peak (AOP) 1200
    8. AOP slope
    9. S_b parameter, b=2.5 (arXiv:1104.3399, arXiv:1305.7439)
    10. S_b parameter, b=4.0
    11. sum of all signals in all detectors
    12. asymmetry  of  the  summed signal  at  the  upper  and  lower layers of detectors
    13. total number of peaks in event
    14. number of peaks for the detector with the largest signal

    ev_ids (num_evs,3):
    0. date
    1. time
    2. particle id (Z for nuclei, 0 for gamma, -1 for real data)

    mc_params (num_evs,10):
    0. mc_event_num
    1. mc_parttype (CORSIKA, 1 - gamma, 14 - proton, 5626 - Fe)
    2. mc_corecounter, closest to core detector number
    3. mc_E (for primaries other than photon energy is rescaled by 1/1.27, i.e. to proton FD energy scale)
    4. mc_theta
    5. mc_phi
    6. mc_height_1st_inter, km
    7. mc_xcore
    8. mc_ycore
    9. mc_border_distance, km 

    dt_params (num_dets,6)
    0. detector x relative to shower core, 1200 m units
    1. detector y relative to shower core, 1200 m units
    2. detector z, 1200 m units
    3. detector signal MIP
    4. time of the plane front arrival, mks
    5. time of the waveform relative to the plane front, mks

    wfs_flat (num_dets,128,2): log(1+wf), float16
    waveforms for all detectors
    0: upper layer signal
    1: lower layer signal

    det_max_wf and det_max_params:
    wfs and its params for the most acive detector

    ev_starts (num_evs):
    For event number i, ev_starts[i] is the first entry in dt_<anything> related to the event, ev_starts[i+1]-1 is the last.
    For example, dt_params[ ev_starts[10]:ev_starts[11] ] corresponds to dt_params for 10th events
    """
    def __init__(self, data_path:str, mode:str, mc_params:bool = False,
                paticles: Optional[List[str]] = None,
                reconstruction_params: Optional[List[int]] = None,
                change_coordinat: bool = False,
                change_sort: bool = False,
                probability = None,
                recos: bool = True,
                make_real_time: bool = True, 
                reduce_grid: bool = True):
        """
        Dataset по нормализованному h5 с переменной длиной события (hits).

        Args:
            data_path: путь к .h5 (группы train/test/val + norm_param).
            mode: сплит для чтения — ``'train'``, ``'test'`` или ``'val'``.
            mc_params: если True, mc_params остаются в пайплайне фильтрации
                (фактически всегда читаются; флаг влияет на choise_def_particles).
            paticles: типы частиц для фильтра, напр. ``['pr', 'photon', 'fe']``;
                None — без фильтра по типу.
            reconstruction_params: индексы колонок mc_params, которые отдаёт
                ``__getitem__`` как params_CR; None — все колонки.
            change_coordinat: сдвинуть x,y детекторов так, чтобы min=0 (сетка от угла).
            change_sort: отсортировать hits по (x, y) перед возвратом.
            probability: доля событий на каждый тип из ``paticles`` (0..1);
                короче списка — дополняется 1.0; None — все 1.0.
            recos: подгружать ли event-level ``recos`` (иначе пустой тензор).
            make_real_time: если True — col4 := нормализованное (t_plane + t_vs_plane),
                col5 отбрасывается → dt_params shape (N, 5); иначе оставляем 6 колонок.
            reduce_grid: Уменьшение шага сетки в 2 раза
        """
        # Запись в генерации h5 file
        self.mass_dict = {'pr': 14,
                     'photon': 1,
                     'fe': 5626}
        self.paticles = paticles
        norm_param, data, ev_starts, mc_params, recos = self.read_h5(
            data_path,
            mode,
            mc_params,
            paticles,
            probability=probability,
            load_recos=recos,
            make_real_time=make_real_time,
        )
        # prepoccessing
        self.norm_param = norm_param
        self.data = data
        self.ev_starts = ev_starts
        self.mc_params = mc_params
        self.recos = recos
        self.reconstruction_params = reconstruction_params
        if reconstruction_params is not None:
            self.reconstruction_params = [int(i) for i in self.reconstruction_params]
        self.change_coordinat = change_coordinat
        self.change_sort = change_sort
        self.reduce_grid = reduce_grid
    def sort_tensor(self, x):
        # Получаем индексы сортировки по первой колонке (col0)
        _, indices_col0 = torch.sort(x[:, 0])  # Сортировка по col0
        sorted_x_col0 = x[indices_col0]  # Сортируем весь тензор по col0

        # Теперь сортируем по второй колонке (col1) с учетом уже отсортированного по col0
        _, indices_col1 = torch.sort(sorted_x_col0[:, 1])  # Сортировка по col1
        sorted_indices = indices_col0[indices_col1]

        # Возвращаем отсортированный тензор
        return x[sorted_indices]

    def __len__(self):
        return len(self.ev_starts)-1
    def preprocc_signal(self, data: np.ndarray, ch: int = 3) -> np.ndarray:
        data[:,ch] = np.log(data[:,ch] - data[:,ch].min()+1e-6)
        return data
    def __getitem__(self, idx):
        st = self.ev_starts[idx]
        fn = self.ev_starts[idx + 1]
        
        mc_params = self.mc_params[idx]
        if self.reconstruction_params:
            params_CR = mc_params[self.reconstruction_params]
        else:
            # all mc params
            params_CR = mc_params
        if self.recos is not None:
            recos = self.recos[idx]
        else:
            recos = np.zeros((0,15))
        x = self.data[st:fn]
        if self.change_coordinat:
            # 0.8027 step
            x_left = x[..., 0].min()
            y_top = x[..., 1].min()
            x[..., 0] = x[..., 0] - x_left
            x[..., 1] = x[..., 1] - y_top
        if self.change_sort:
            x=self.sort_tensor(x)
        if self.reduce_grid:
            x_SR = self.find_closes__detector(x)
        else:
            x_SR = x
        return torch.tensor(x), torch.tensor(mc_params[1]), torch.tensor(params_CR), torch.tensor(recos), torch.tensor(x_SR)
    def read_h5(self, data_path, mode, mc_params, paticles: Optional[List[str]] = None, 
                probability: List[float] = None, load_recos: bool = False, make_real_time: bool = True):
        """
        Читает .h5 файл, выбирает нужный режим и фильтрует события по частицам.
        load_recos: подгрузить train['recos'] и вернуть строки только для отфильтрованных событий.
        """
        if probability is None:
            probability = [1.0] * len(paticles)
        probability = list(probability) + [1.0] * (len(paticles) - len(probability))
        with h5.File(data_path,'r') as f:
            if mode not in f:
                raise KeyError(f"В файле нет группы '{mode}'. Доступные ключи: {list(f.keys())}")
            train = f[mode]
            dt_params = torch.tensor(train['dt_params'][()])
            ev_starts = torch.tensor(train['ev_starts'][()])
            recos = torch.tensor(train['recos'][()])
            # Копируем в память: после выхода из with файл закрыт, h5-группы невалидны
            norm_param_mean = torch.tensor(f['norm_param']['dt_params']['mean'][()])
            norm_param_std = torch.tensor(f['norm_param']['dt_params']['std'][()])
            norm_param = {
                'dt_params': {
                    'mean': norm_param_mean,
                    'std': norm_param_std,
                }
            }
            if mc_params:
                mc_params = torch.tensor(train['mc_params'][()])
                if paticles is not None:
                    dt_params, ev_starts, mc_params, recos = self.choise_def_particles(paticles, data = dt_params, ev_starts=ev_starts, mc_params=mc_params, recos=recos, get_mc_params=True,
                                                    probability = probability)
            
            else:
                mc_params = torch.tensor(train['mc_params'][()])
                if paticles is not None:
                    dt_params, ev_starts, recos = self.choise_def_particles(paticles, data = dt_params, ev_starts=ev_starts, mc_params=mc_params, recos=recos, get_mc_params=False,
                                            probability = probability)
            # mc_params = None
        print('data in read_h5: dt_params, ev_starts, mc_params, recos', dt_params.shape, ev_starts.shape, mc_params.shape, recos.shape)
        if make_real_time:
            flat = dt_params[:,4] * norm_param_std[4] + norm_param_mean[4]
            diff_time = dt_params[:,5] * norm_param_std[5] + norm_param_mean[5]

            Time = flat + diff_time
            mean_Time = Time.mean()
            std_Time = Time.std()
            Time = (Time - mean_Time) / std_Time
            dt_params[:,4] = Time
            dt_params = dt_params[:,:5]
        print(dt_params.shape)
        return norm_param, dt_params, ev_starts, mc_params, recos
    def str2mass(self, name: List[str]) -> List[int]:
        #1. mc_parttype (CORSIKA, 1 - gamma, 14 - proton, 5626 - Fe)
        mass_dict = self.mass_dict
        return [mass_dict[n] for n in name]
    def choise_def_particles(self, name: List[str], data, ev_starts, mc_params, recos, par_num: int = 1, get_mc_params: bool = False, 
                            probability: List[float] = None):
        """
        Фильтрует события по типам частиц.

        Аргументы:
            name (List[str]): Список имён частиц (например, ['photon']).
            data (Tensor): Все данные по событиям.
            ev_starts (Tensor): Индексы начала событий.
            mc_params (Tensor): Метаданные событий.
            par_num (int): Индекс параметра с массой частицы.
            get_mc_params (bool): Вернуть ли отфильтрованные mc_params.
            probability (List[float]): вероятность для каждого типа частицы (0..1).
                Если короче len(name), недостающие дополняются 1.0. По умолчанию [1.0]*len(name).

        Возвращает:
            Tuple[Tensor, Tensor, Optional[Tensor]]: Отфильтрованные данные, ev_starts, опционально — mc_params.
        """
        if probability is None:
            probability = [1.0] * len(name)
        probability = list(probability) + [1.0] * (len(name) - len(probability))

        # Преобразуем имена частиц в массы
        mass = self.str2mass(name)
        # Булева маска для корректной индексации (не 0/1!)
        mask = torch.zeros(mc_params.shape[0], dtype=torch.bool, device=mc_params.device)
        for i, m in enumerate(mass):
            mask_per_mass = torch.isin(mc_params[:, par_num], torch.tensor([m], device=mc_params.device))
            rand_vals = torch.rand(mask_per_mass.shape[0], device=mc_params.device)
            probability_mask = rand_vals < probability[i]
            mask = mask | (mask_per_mass & probability_mask)
        # Применяем маску к mc_params и ev_starts
        mc_params_filtered = mc_params[mask]
        recos_filtered = recos[mask]
        # Вычисляем новые индексы начала событий
        ev_starts_new = torch.cat([torch.tensor([0], device=data.device), torch.cumsum(ev_starts[1:][mask] - ev_starts[:-1][mask], dim=0)])  
        # Собираем данные, соответствующие выбранным событиям
        try:
            data_indices = torch.cat([torch.arange(ev_starts[i], ev_starts[i+1], device=data.device) for i in torch.where(mask)[0]])
            data_new = data[data_indices]
        except NotImplementedError:
            raise NotImplementedError("Не найдены события с заданными частицами.")
        if get_mc_params:
            return data_new, ev_starts_new, mc_params_filtered, recos_filtered
        else:
            return data_new, ev_starts_new, recos_filtered
    def cut_ev_start(self, name: List[str], ev_starts, mc_params, par_num: int = 1, get_mc_params: bool = False):
        mass = self.str2mass(name)
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
    def find_closes__detector(self, dt_params):
        '''
        dt_params - shape (N, 5) или 6 если плоский фронт
        '''
        norm_param_dt_params = self.norm_param['dt_params']
        norm_param_dt_params_mean = norm_param_dt_params['mean'][:2]
        norm_param_dt_params_std = norm_param_dt_params['std'][:2]



        coordinates = dt_params[:, :2]
        # делаем переномировку 
        coordinates = coordinates*norm_param_dt_params_std.unsqueeze(0) + norm_param_dt_params_mean.unsqueeze(0)
        distant = torch.sum(torch.pow(coordinates, 2), dim =1) # shape N
        argmin = torch.argmin(distant) # индекс самого ближайшего детектора
        # print('argmin', argmin)
        coordinates_from_armin = coordinates - coordinates[argmin] # координаты относительно самого ближнего
        # шаг сетки относительно постоянный и равен около 
        # print('coordinates_from_armin', coordinates_from_armin)
        coordinates_from_armin = torch.round(coordinates_from_armin)

        # print('coordinates_from_armin', coordinates_from_armin)
        index = torch.where((coordinates_from_armin[:,0]%2==0) * (coordinates_from_armin[:,1]%2==0))
        # print(index)
        # print(coordinates[index])
        return dt_params[index]

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
    make_real_time = config['make_real_time']
    if isinstance(start_token,int):
        start_token = torch.ones((1,6)) * start_token
    elif isinstance(start_token, str):
    
        if start_token == 'hard_v1':
            # TODO: change  
            # signal min -0.277908
            # flat min -8.798042
            # real-flat max 15.595449
            if make_real_time:
                start_token = torch.tensor([0, 0, 0, -0.277908, -8.798042+15.595449]).unsqueeze(0)
            else:
                start_token = torch.tensor([0, 0, 0, -0.277908, -8.798042, 15.595449]).unsqueeze(0)
        elif start_token == 'trainable':
            raise ValueError("Пока решил реализовывать CLS token, отличный от страрт")
        else:
            raise ValueError(f"Unknown start_token: {start_token}")
    else:
        raise ValueError(f"Unknown start_token: {start_token}")

    if make_real_time:
        stop_token = torch.ones((1,5)) * stop_token
    else:
        stop_token = torch.ones((1,6)) * stop_token
    return {'start_token': start_token, 'stop_token': stop_token, 'padding_value': padding_value}
def collate_fn_many_args(batch: Tensor, start_token: Tensor, stop_token: Tensor, padding_value: int, mc_params: bool = False, 
                        th_num_det: Optional[int] = None) -> Tensor:
    #, padding_value, star_token, stop_toke
    """
    Кастомная функция для DataLoader, которая заполняет последовательности до максимальной длины в батче.
    th_num_det - порог по кол-ву сработавших дететоров. нижний порог 

    Предается много аргументов. Поэтому преед использованием в DataLoader надо сделать wrapper_mask(collate_fn_many_args, ...)
    """
    # start_token, stop_token = torch.zeros((1,6)), torch.zeros((1,6))
    # Извлекаем каждую последовательность в батче
    # for i in range(len(batch)):
    #     print(i, batch[i])
    if mc_params:
        sequences = []
        params = None
        particles = None
        recos = None
        sequences_SR = []
        for item, part, par, recos_, x_SR in batch:
            if th_num_det is not None:
                if x_SR.shape[0]< th_num_det:
                    # пропускаем, слишком мало детекторов
                    continue
            sequences.append(torch.cat((start_token, item, stop_token), dim=0))
            sequences_SR.append(torch.cat((start_token, x_SR, stop_token), dim=0))
            if params is None:
                params = par.unsqueeze(0)
                particles = part.unsqueeze(0)
                recos = recos_.unsqueeze(0)
            else:
                params = np.concatenate((params, par.unsqueeze(0)), axis=0)
                particles = np.concatenate((particles, part.unsqueeze(0)), axis=0) 
                recos = np.concatenate((recos, recos_.unsqueeze(0)), axis=0)
        # Заполняем последовательности до одинаковой длины
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value)  # (batch_size, max_seq_len, 5)
        padded_sequences_SR = pad_sequence(sequences_SR, batch_first=True, padding_value=padding_value)  # (batch_size, max_seq_len, 5)
        # padded_sequences_SR бля эта ебатория для уменьшения размерности. Она по shape вообще не должна совпадать с X. Потому что енкодер и так выдаст унифицированных латеное предстваление
        return padded_sequences, torch.tensor(particles), torch.tensor(params), torch.tensor(recos), padded_sequences_SR
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

    # Used and need

    # padding_value = kwargs['padding_value']
    # start_token = kwargs['start_token']
    # stop_token = kwargs['stop_token']
    def wrapper_func(batch):
        return func(batch, **kwargs)
    return wrapper_func
def test_h5_file(data_path:str):
    '''
    Показывает что внутри у h5 файла. Классическая структура указана наверху в докстринге.
    '''
    print(f'------start test {data_path}-----\n')
    with h5.File(data_path,'r') as f:
        print(f.keys())
        for d in f['train'].keys():
            shape = f['train'][d].shape
            print(f'key: {d} sahpe {shape}')
        ev_start = f['train']['ev_starts']
        num_det = np.diff(ev_start)
        print(num_det.mean(), num_det.std())
        plt.hist(num_det, log=True, histtype='step', label = os.path.basename(data_path))
        norm_param = f['norm_param']
        print('norm_param', norm_param.keys())
        norm_param_dt_params = norm_param['dt_params']
        print('norm_param_dt_params', norm_param_dt_params.keys(), norm_param_dt_params["mean"].shape)
    print(f'------end test {data_path}------\n')

if __name__ == '__main__':
    print("-----TEST VariableLengthDataset------\n")

    print("------- test -----\n")
    import matplotlib.pyplot as plt
    import os
    data_path1 = '/home3/rfit/Telescope_Array/phd_work/data/normed/prhenife_q4_1745_14yr_0110_bundled_one_work_rubsov_approx.h5'
    data_path ='/home3/rfit/Telescope_Array/phd_work/data/normed/pr_q4_1895_no_sat_no_geo_0110_bundled_one_work_rubsov_approx.h5'
    # тут все работало
    data_old = '/home3/rfit/Telescope_Array/phd_work/data/normed/Ivan_Kharuk_pr_ga_all_0001_eq_eff_normed_one_work.h5'
    fig = plt.figure()
    test_h5_file(data_path=data_path)

    test_h5_file(data_path=data_path1)
    test_h5_file(data_path=data_old)
    plt.legend()
    plt.grid()
    plt.xlabel('Num. det.')
    plt.ylabel('LOG Density')
    plt.savefig('detecors.png')
    dataset = VariableLengthDataset(
        data_path=data_path, 
        mode = 'train',
        mc_params=True,
        make_real_time=True,
        paticles = ['pr']
    )
    for data_unit in dataset:
        print(data_unit)
        break