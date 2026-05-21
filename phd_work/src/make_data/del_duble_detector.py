import h5py as h5
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def get_length(x):
    return len(x) - np.sum(x[:, 0] == -11) - 2

def double_detector(x):
    length = get_length(x)
    coords = x[:, :4]

    # Получаем уникальные строки и индексы их появления
    unique_coords, inverse_indices, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)

    # Находим координаты-дубликаты
    duplicate_coords = unique_coords[counts > 1]

    # Находим индексы всех дубликатов в оригинальном массиве
    duplicate_indices = []
    for coord in duplicate_coords:
        matches = np.all(coords == coord, axis=1)
        indices = np.nonzero(matches)[0].tolist()
        if len(indices) > 1:
            duplicate_indices.append(indices)

    return duplicate_indices

def remove(x, double_index: list):
    all_index = np.ones(len(x)).astype(bool)
    
    # Обрабатываем каждую группу дубликатов
    for group in double_index:
        if not group:
            continue
            
        # Находим индекс строки с минимальным значением в последнем столбце
        double = x[group]
        min_in_double = np.argmin(double[:,-1])
        save_index = group[min_in_double]
        group.pop(min_in_double)
        all_index[group] = False
    result = x[all_index]      
    return result
def get_data(dt_params, ev_starts): 
    data = []
    start = ev_starts[0] # it's 0
    for ind in tqdm(ev_starts[1:]):
        data.append(dt_params[start:ind])
        start = ind
    return data
def get_new_data(data):
    data_new = []
    for i in tqdm(data):
        index = double_detector(i)
        res = remove(i, index)    
        data_new.append(res)
    ev_starts_new = [0]
    dt_params_new = np.concatenate(data_new, axis=0)
    for d in tqdm(data_new):
        # dt_params_new = np.concatenate((dt_params_new, d), axis=0)
        ev_starts_new.append(len(d) + ev_starts_new[-1])
    ev_starts_new = np.array(ev_starts_new)
    return dt_params_new, ev_starts_new


def main(h5_in, h5_out):
    with h5.File(h5_in,'r') as f, h5.File(h5_out, 'w') as ho:
        # Сначала копируем весь вход в выход. Иначе при 'w' файл сразу пустой:
        # любая ошибка в get_new_data до старого места цикла copy оставляла бы пустой h5.
        for item_name in f:
            f.copy(item_name, ho)

        mean = np.array(f['norm_param']['dt_params']['mean'][()])
        std = np.array(f['norm_param']['dt_params']['std'][()])
        print(mean, std)
        train = f['train']
        dt_params = np.array(train['dt_params'][()])
        ev_starts = np.array(train['ev_starts'][()])

        test = f['test']
        dt_params_test = np.array(test['dt_params'][()])
        ev_starts_test = np.array(test['ev_starts'][()])

        data = get_data(dt_params, ev_starts)
        data_test = get_data(dt_params_test, ev_starts_test)

        dt_params_new, ev_starts_new = get_new_data(data)
        dt_params_new_test, ev_starts_new_test = get_new_data(data_test)

        # Заменяем dt_params в тренировочной группе
        if 'train' in ho and 'dt_params' in ho['train']:
            # Удаляем старый набор данных
            del ho['train']['dt_params']
            del ho['train']['ev_starts']
            # Создаем новый набор с обновленными данными
            ho['train'].create_dataset('dt_params', data=dt_params_new)
            ho['train'].create_dataset('ev_starts', data=ev_starts_new)
        if 'test' in ho and 'dt_params' in ho['test']:
            # Удаляем старый набор данных
            del ho['test']['dt_params']
            del ho['test']['ev_starts']
            # Создаем новый набор с обновленными данными
            ho['test'].create_dataset('dt_params', data=dt_params_new_test)
            ho['test'].create_dataset('ev_starts', data=ev_starts_new_test)
def test(h5_in, h5_out):
    """Сравнивает структуру входа и результата main(). h5_out только для чтения ('r')."""
    print('--- вход (h5_in) ---')
    with h5.File(h5_in, 'r') as f:
        print(f.keys())
        train = f['train']
        for key in train.keys():
            print('train', key, np.array(train[key][()]).shape)
        test = f['test']
        for key in test.keys():
            print('test', key, np.array(test[key][()]).shape)
    print('--- выход (h5_out после main) ---')
    with h5.File(h5_out, 'r') as ho:
        print(ho.keys())
        train = ho['train']
        for key in train.keys():
            print('train', key, np.array(train[key][()]).shape)
        test = ho['test']
        for key in test.keys():
            print('test', key, np.array(test[key][()]).shape)
if __name__ == '__main__':

    h5_out = '/home3/rfit/Telescope_Array/phd_work/data/normed/Ivan_Kharuk_pr_ga_all_0001_eq_eff_normed_one_work.h5'
    h5_in = '/home3/ivkhar/TA/data/MC/normed/gamma_search/pr_ga_all_0001_eq_eff_normed.h5'
    main(h5_in, h5_out)
    test(h5_in, h5_out)
