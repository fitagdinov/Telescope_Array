import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
import h5py as h5
from train_VAE import datasets
import torch
# import train_VAE.datasets as datasets
# import torch
from train_VAE.utils import read_config
config = read_config('config.yaml')
# Загрузка датасета из h5-файла.
with h5.File(config['data_path'],'r') as f:
    print('keys', list(f.keys()))
    train = f['train']
    dt_params = torch.tensor(train['dt_params'][()])
    ev_starts = torch.tensor(train['ev_starts'][()])
    mc_params = torch.tensor(train['mc_params'][()])
# Выберем 5 первых событий с протоном 
k=0
k_stop = 5
ev_starts_def_par = []
for i,par in enumerate(mc_params):
    # Выбираем события с протоном (mass=14) и добавляем кортежи в список.
    if par[1] == 5626:
        ev_starts_def_par.append((ev_starts[i], ev_starts[i+1]))
        k += 1
        if k == k_stop:
            break
# Проходимся по старотовому и конечному индесу события и считываем данные 
data = []
for st,fn in ev_starts_def_par:
    data.append(dt_params[st:fn])

# Загрузка датасета из модуля в обучении.
# Берем для примера протон
dataset = datasets.VariableLengthDataset(config['data_path'], 'train', paticles=['fe'])

# Проверяем что наше чтение файлов действительно коректно. Вывод должен быть равен 0
for k in range(k_stop):
    print(torch.max(data[k] - dataset[k]))
