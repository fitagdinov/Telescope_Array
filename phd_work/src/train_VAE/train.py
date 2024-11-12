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

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(filename_suffix = 'c_in_encoder')

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
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=-1.0)  # (batch_size, max_seq_len, 5)
    return padded_sequences

# Пример данных (можете заменить его своими реальными данными)
# Создадим случайные данные с длиной от 5 до 120 для каждой последовательности
data = [torch.randn(torch.randint(5, 121, (1,)).item(), 5) for _ in range(100)]  # 100 последовательностей разной длины
# Инициализируем датасет и DataLoader
dataset = VariableLengthDataset(data_path, 'train')
val_dataset = VariableLengthDataset(data_path, 'test')
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, hidden_dim2):
        super(Encoder, self).__init__()
        hidden_dim_1, hidden_dim_2, hidden_dim_3 = 32, 64, 128 
        # self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm_2 = nn.LSTM(input_dim, hidden_dim_1, batch_first=True)
        self.lstm_3 = nn.LSTM(hidden_dim_1, hidden_dim_2, batch_first=True)
        self.lstm_4 = nn.LSTM(hidden_dim_2, hidden_dim_3, batch_first=True)
        self.add_fc = nn.Linear(hidden_dim_3, hidden_dim2)
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)

    def forward(self, x):
        # TODO ЧТО ЭТО ТАКОЕ И КАК С ЭТИМ РАБОТАТЬ, ЧТО НАМ НУЖНО
        x, (h_n, c_n) = self.lstm_2(x)  # h_n shape: (1, batch_size, hidden_dim)
        x, (h_n, c_n) = self.lstm_3(x) 
        x, (h_n, c_n) = self.lstm_4(x) 
        # TODO дописать маску и читать аутпут[13]
        h_n = c_n.squeeze(0)[-1]  # убираем первую размерность
        h_n = torch.relu(h_n)
        mu = self.fc_mu(h_n)  # среднее латентного пространства
        log_var = self.fc_logvar(h_n)  # логарифм дисперсии латентного пространства
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, hidden_dim2):
        super(Decoder, self).__init__()
        hidden_dim_2, hidden_dim_3 = 128, 200
        self.fc = nn.Linear(latent_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim_2, batch_first=True)
        
        self.lstm2 = nn.LSTM(hidden_dim_2, hidden_dim_3, batch_first=True)
        
        self.lstm3 = nn.LSTM(hidden_dim_3, hidden_dim, batch_first=True)
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, seq_len):
        h = torch.relu(self.fc(z)) # (batch_size, 1, hidden_dim)
        h = torch.relu(self.fc2(h)).unsqueeze(1)
        h = h.repeat(1, seq_len, 1)  # Повторяем скрытое состояние для каждого шага времени
        lstm_out, _ = self.lstm(h)  # Проходим через LSTM
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out, _ = self.lstm3(lstm_out)
        return self.output_layer(lstm_out)

class VAE(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, latent_dim=16, hidden_dim2=64):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, hidden_dim2)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, hidden_dim2)
        print("Encoder has params:", self.count_parameters(self.encoder),"Decoder has params:", self.count_parameters(self.decoder))
        

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        seq_len = x.size(1)
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z, seq_len)
        return recon_x, mu, log_var
    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)

# Функция потерь
def vae_loss(recon_x, x, mu, log_var):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_divergence / x.size(0)

# Настройка обучения
input_dim = 6
hidden_dim = 64
latent_dim = 16
batch_size = 400
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
model = VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# Цикл обучения (предполагается, что train_loader предоставляет пакеты последовательностей переменной длины)
epochs = 20
PATH = '../../Models_VAE/model_cn_ADD_FC_MORE_LSTM'
os.makedirs(PATH, exist_ok = True)
iters = 0
for epoch in range(epochs):
    model.train()
    pbar = tqdm(train_loader, desc =f"Epoch {epoch + 1}/{epochs}, Loss: 0.0")
    pbar_val = tqdm(val_loader, desc =f"Epoch {epoch + 1}/{epochs}, Loss: 0.0")
    for x in pbar:  # x должен быть пакетом последовательностей с заполнением
        x = x.to(device)
        optimizer.zero_grad()
        recon_x, mu, log_var = model(x)
        loss = vae_loss(recon_x, x, mu, log_var)
        writer.add_scalar("Loss/train", loss, iters)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        iters += 1
    model.eval()
    loss_mean = []
    for x in pbar_val:  # x должен быть пакетом последовательностей с заполнением
        x = x.to(device)
        recon_x, mu, log_var = model(x)
        loss = vae_loss(recon_x, x, mu, log_var)
        loss_mean.append(loss.item())
        pbar_val.set_description(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), os.path.join(PATH, f'epoch_{epoch}'))
    val_loss = np.array(loss_mean).mean()
    print(f'Epoch {epoch + 1}, Loss: {val_loss}')
    writer.add_scalar("Loss/val", val_loss, epoch)