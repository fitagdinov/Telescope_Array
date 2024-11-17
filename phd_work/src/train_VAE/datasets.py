import torch
from torch.utils.data import Dataset, DataLoader
import h5py as h5
from torch.nn.utils.rnn import pad_sequence

class VariableLengthDataset(Dataset):
    def __init__(self, data_path, mode):
        """
        Args:
            data: список тензоров, где каждый тензор имеет форму (seq_len, 5)
        """
        data, ev_starts = self.read_h5(data_path, mode)
        self.data = data
        self.ev_starts = ev_starts

    def __len__(self):
        return len(self.ev_starts)-1

    def __getitem__(self, idx):
        st = self.ev_starts[idx]
        fn = self.ev_starts[idx + 1]
        return torch.tensor(self.data[st:fn])
    def read_h5(self, data_path, mode):
        with h5.File(data_path,'r') as f:
            print('keys', list(f.keys()))
            train = f[mode]
            # norm_param = f['norm_param']['dt_params']
            # norm_param_std = norm_param['std'][()]
            # norm_param_mean = norm_param['mean'][()]
            # print(list(norm_param.keys()))
            # print(list(train.keys()))
            dt_params = train['dt_params'][()]
            ev_starts = train['ev_starts'][()]
        return dt_params, ev_starts
        

def collate_fn(batch):
    """
    Кастомная функция для DataLoader, которая заполняет последовательности до максимальной длины в батче.
    """
    # Извлекаем каждую последовательность в батче
    sequences = [item for item in batch]
    # Заполняем последовательности до одинаковой длины
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=-10.0)  # (batch_size, max_seq_len, 5)
    return padded_sequences