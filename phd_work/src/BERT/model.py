from torch import nn
import torch 
from torch import Tensor
from typing import Optional
import torch
import math
class Encoder_Transformer(nn.Module):
    """
    Энкодер вариационного автокодировщика на базе Transformer.
    За основу классификатор частиц (См. particle_classification/classification_models.py).

    Аргументы:
        input_dim (int): Размерность входных данных.
        hidden_dim (int): Размерность скрытого состояния LSTM.
        latent_dim (int): Размерность латентного пространства.
        lstm2 (bool): Добавлять ли второй LSTM-слой.
        lstm3 (bool): Добавлять ли третий LSTM-слой.
    """
    def __init__(self, input_dim=6, hidden_dim=64, latent_dim=16,num_layers=4, **kwargs):
        super().__init__()
        self.embading  = nn.Linear(input_dim,hidden_dim)
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                                nhead = 4,
                                                                dim_feedforward=1024,
                                                                dropout=0.1,
                                                                activation='relu',
                                                                layer_norm_eps=1e-05, 
                                                                batch_first=True, 
                                                                norm_first=False, 
                                                                )
        self.TransformerEncoder = nn.TransformerEncoder(
                                    self.TransformerEncoderLayer,
                                    num_layers=num_layers,
                                                        )
        self.config = kwargs
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.stop_token = kwargs['stop_token']
        self.padding_value = kwargs['padding_value']
        self.fc2 = nn.Linear(32, latent_dim)
        
        # Слои для VAE: mu и log_var
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # self.softmax = nn.Softmax(dim=1)
        self.activation = nn.LeakyReLU()
    def get_mask(self, x, stop_token = None, padding_value = None):
        if stop_token is None:
            stop_token = torch.tensor(self.stop_token, dtype=torch.long, device=x.device)
        if padding_value is None:
            padding_value = torch.tensor(self.padding_value, dtype=torch.long, device=x.device)
        mask = torch.zeros_like(x, dtype=torch.long)  # Убедитесь, что это long (int64)
        mask = torch.where(x == stop_token, torch.tensor(1, dtype=torch.long, device=x.device), mask)
        mask = torch.where(x == padding_value, torch.tensor(1, dtype=torch.long, device=x.device), mask)

        # ОСОВОБОДИМ ПЕРВЫЙ ТОКЕН. ОН БУДЕТ CLS в пониманиие БЕРТ.
        # ПО нему и будем постанавливать. Он будет агрегировать в СЕбе все

        # mask[:,0,:] = 1
        mask = mask[:,:,0] # need (batch, seq)

        # unused MASK. BE  carefull
        return mask.bool().to(x.device)


    def forward(self,x):
        mask = self.get_mask(x)
        
        x = self.embading(x)
        x = self.TransformerEncoder(x, src_key_padding_mask= mask)
        # Только по этой оси потому что она переменной длины
        CLS = x[:,0,:]  
    
        # CLS = torch.mean(x, dim=1)
        # print(x.shape, mask.shape)
        # CLS = torch.sum(x*(~mask.unsqueeze(-1)), dim=1)
        # CLS = CLS/(torch.sum(~mask, dim=1).unsqueeze(-1))

        # mu, log_var, (h_n, c_n) = self.last_encoder(x)
        # z=mu


        # Общий слой для извлечения признаков
        z = self.fc1(CLS)
        z = self.activation(z)
        
        # Вычисляем mu и log_var для VAE
        mu = self.fc_mu(z)
        # для соблюдения выхода как у LSTM
        return mu
    def load(self, path):
        if path is not None:
            self.load_state_dict(torch.load(path))
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:x.size(1), :]

class EncoderTransformerMask(nn.Module):
    """
    Энкодер вариационного автокодировщика на базе Transformer.
    За основу классификатор частиц (См. particle_classification/classification_models.py).

    Аргументы:
        input_dim (int): Размерность входных данных.
        hidden_dim (int): Размерность скрытого состояния LSTM.
        latent_dim (int): Размерность латентного пространства.
        lstm2 (bool): Добавлять ли второй LSTM-слой.
        lstm3 (bool): Добавлять ли третий LSTM-слой.
    """
    def __init__(self, input_dim=6, hidden_dim=64, latent_dim=16,num_layers=4, 
                        max_seq_len=100, **kwargs):
        super().__init__()
        self.embading  = nn.Linear(input_dim,hidden_dim)
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                                nhead = 2,
                                                                dim_feedforward=hidden_dim*4,
                                                                dropout=0.2,
                                                                activation='relu',
                                                                layer_norm_eps=1e-05, 
                                                                batch_first=True, 
                                                                norm_first=False,
                                                                )
        self.TransformerEncoder = nn.TransformerEncoder(
                                    self.TransformerEncoderLayer,
                                    num_layers=num_layers,
                                                        )
        self.pos_embedding = PositionalEncoding(hidden_dim, max_len=max_seq_len)
        self.config = kwargs
        self.fc1 = nn.Linear(hidden_dim, input_dim)
        self.stop_token = kwargs['stop_token']
        self.padding_value = kwargs['padding_value']
        # self.fc2 = nn.Linear(32, latent_dim)
        self.activation = nn.LeakyReLU()
        self.hidden_dim = hidden_dim
        self.MASK_TOKEN = torch.nn.Parameter(torch.tensor([[[ 0.7345, -0.4238, -0.3467, -0.7196,  0.4177, -0.6103]]]))
        # print("used_MAE", kwargs["used_MAE"])
        if "used_MAE" in kwargs.keys() and kwargs["used_MAE"]:
            self.embading_2 = nn.Linear(hidden_dim,hidden_dim)
            # self.embading_3 = nn.Linear(2*hidden_dim,hidden_dim)
            # self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
            # self.batchnorm2 = nn.BatchNorm1d(2*hidden_dim)
            self.dropout = nn.Dropout(p=0.3)

            # Инициализируем MASK_TOKEN более консервативно
            self.MASK_TOKEN = torch.nn.Parameter(torch.zeros(1,1,hidden_dim))
            self.used_MAE = kwargs["used_MAE"]
            
            # Инициализируем веса для предотвращения проблем с градиентами
            self._init_weights()
        else:
            self.used_MAE = False
        # self.fc_latent_space = nn.Linear(hidden_dim, latent_dim)
        self.latent_dim = latent_dim
    def get_mask(self, x, stop_token = None, padding_value = None):
        if stop_token is None:
            stop_token = torch.tensor(self.stop_token, dtype=torch.long, device=x.device)
        if padding_value is None:
            padding_value = torch.tensor(self.padding_value, dtype=torch.long, device=x.device)
        mask = torch.zeros_like(x, dtype=torch.long)  # Убедитесь, что это long (int64)
        mask = torch.where(x == stop_token, torch.tensor(1, dtype=torch.long, device=x.device), mask)
        mask = torch.where(x == padding_value, torch.tensor(1, dtype=torch.long, device=x.device), mask)

        # ОСОВОБОДИМ ПЕРВЫЙ ТОКЕН. ОН БУДЕТ CLS в пониманиие БЕРТ.
        # ПО нему и будем постанавливать. Он будет агрегировать в СЕбе все

        # mask[:,0,:] = 0

        mask = mask[:,:,0] # need (batch, seq)

        # unused MASK. BE  carefull
        return mask.bool().to(x.device)

    def _init_weights(self):
        """Инициализация весов для предотвращения проблем с градиентами"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot инициализация
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self,x:Tensor, masking:Tensor):
        # mask like in global mean of task
        # masking -> batch, lenght, chanal

        mask_tensor = torch.repeat_interleave(self.MASK_TOKEN, x.size(0), dim=0)
        mask_tensor = torch.repeat_interleave(mask_tensor, x.size(1), dim=1)
        mask = self.get_mask(x) # макска на вспомогательные токены (НЕ НА МАСКИРОВОЧНЫЕ)
        
        if self.used_MAE:
            # TODO (екодер н етрогает токен  стр 124-132 выше)
            # Сохраняем координаты для позиционной информации
            coordinates = x[:, :, :3]
            # Простой эмбеддинг с регуляризацией
            x = self.embading(x)
            x = self.activation(x)
            x = self.dropout(x)
            
            # Дополнительный слой с batch normalization
            x = self.embading_2(x)
            x = self.activation(x)
            x = self.dropout(x)
            x[:,:,:3] = coordinates
            # Применяем маскирование ДО эмбеддинга
            masking = torch.repeat_interleave(masking[:,:,0:1], x.size(-1), dim=-1)
            # x_masked = torch.where(masking_expanded, mask_tensor[:, :, :x.size(-1)], x)
            masking[:, :, :3] = False # координаты не маскируем.
            x_masked = torch.where(masking, mask_tensor, x) # где маска - вставляем маску, где нет - данные
            
            
            
            # Восстанавливаем координаты в эмбеддинге (первые 3 измерения)
            x[:, :, :3] = coordinates
        else:
            # MASK token пропускаем через эмбэдинг
            #coordinat need for possition. they are first 3 dimmentions
            masking[:, :, :3] = False # координаты не маскируем.
            x_masked = torch.where(masking, mask_tensor, x) # где маска - вставляем маску, где нет - данные
            x = self.embading(x_masked)
        x = self.pos_embedding(x)
        x = self.TransformerEncoder(x, src_key_padding_mask= mask)

        z = self.fc1(x)

        CLS_token = x[:,0,:]
        # latent = self.fc_latent_space(CLS_token)
        # x[:,0, :] -> batch, 64 -> только оно и обновляется при 
        # lat = x.mean(dim=1)
        # lat = concat (CLS, lat) -> Decoder 

        # self.lat = nn.Linear(hid, lat_dim)
        # lat = self.lat(x[:,0, :])
        return z, CLS_token
    def load(self, path):
        if path is not None:
            print('load model', path)
            self.load_state_dict(torch.load(path), strict=True)

if __name__ == '__main__':
    model = EncoderTransformerMask(stop_token = -10, padding_value=-11)
    x =torch.randn(1,30,6)
    masking = torch.ones_like(x).to(torch.bool)
    masking[:,0] = False
    masking[:,20:] = False
    recon = model(x, masking)
    print('recon', recon)
