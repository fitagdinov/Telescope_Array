from torch import nn
import torch 
from torch import Tensor
from typing import Optional
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from logging import logging
# loger = logging.getLogger(__name__)

class Encoder(nn.Module):
    """
    Энкодер вариационного автокодировщика на базе LSTM.

    Аргументы:
        input_dim (int): Размерность входных данных.
        hidden_dim (int): Размерность скрытого состояния LSTM.
        latent_dim (int): Размерность латентного пространства.
        lstm2 (bool): Добавлять ли второй LSTM-слой.
        lstm3 (bool): Добавлять ли третий LSTM-слой.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, lstm2: bool = False, lstm3: bool = False):
        super(Encoder, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim*2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim*2, latent_dim)
        self.lstm2 = None
        self.lstm3 = None
        if lstm2:
            self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        if lstm3:
            self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
    def seq_LSTMs(self, x, lengths=None):
        """
        Последовательно применяет до трёх LSTM-слоёв.

        Аргументы:
            x (Tensor): Последовательность входных данных.
            lengths (Tensor, optional): Длины последовательностей для pack_padded_sequence.

        Возвращает:
            Tuple[Tensor, Tensor]: Последний выход и скрытое состояние.
        """
        # Упаковываем последовательности, чтобы игнорировать padding
        if lengths is not None:
            x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            h_packed, c_n = self.lstm1(x_packed)
            h, _ = pad_packed_sequence(h_packed, batch_first=True)
        else:
            h, c_n = self.lstm1(x)
        
        if self.lstm2 is not None:
            if lengths is not None:
                h_packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
                h_packed, c_n = self.lstm2(h_packed)
                h, _ = pad_packed_sequence(h_packed, batch_first=True)
            else:
                h, c_n = self.lstm2(h)
        
        if self.lstm3 is not None:
            if lengths is not None:
                h_packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
                h_packed, c_n = self.lstm3(h_packed)
                h, _ = pad_packed_sequence(h_packed, batch_first=True)
            else:
                h, c_n = self.lstm3(h)
        return h, c_n

    def forward(self, x, lengths=None):
        """
        Прямой проход энкодера.

        Аргументы:
            x (Tensor): Входная последовательность формы (batch, seq_len, input_dim).
            lengths (Tensor, optional): Длины последовательностей (без учета padding).

        Возвращает:
            mu (Tensor): Вектор средних латентного распределения.
            log_var (Tensor): Логарифм дисперсий латентного распределения.
            Tuple[Tensor, Tensor]: Последние состояния h_n и c_n.
        """
        h, c_n = self.seq_LSTMs(x, lengths=lengths)  # h_n shape: (1, batch_size, hidden_dim)
        # for LSTM
        # Берем последний валидный элемент для каждой последовательности
        if lengths is not None:
            # lengths - это тензор с длинами, нужно получить последний валидный элемент для каждого батча
            batch_size = h.size(0)
            last_indices = (lengths - 1).clamp(min=0)  # индексы последних валидных элементов
            h_n = h[torch.arange(batch_size, device=h.device), last_indices]  # (batch, hidden_dim)
            # Для c_n берем последний элемент из последнего слоя
            c_n_last = c_n[0][-1]  # последний слой, (batch, hidden_dim)
            c_n_cell = c_n[1][-1]  # последний слой, (batch, hidden_dim)
            c_n_concat = torch.concat((c_n_last, c_n_cell), dim=1)  # (batch, hidden_dim*2)
        else:
            # Если lengths не предоставлены, используем последний элемент
            c_n = torch.concat((c_n[0], c_n[1]), dim=2)
            h_n = c_n.squeeze(0)  # убираем первую размерность
            c_n_concat = h_n
        
        mu = self.fc_mu(c_n_concat)  # среднее латентного пространства
        log_var = self.fc_logvar(c_n_concat)  # логарифм дисперсии латентного пространства
        return mu, log_var, (h_n, c_n)
class Encoder_Transformer_AE(nn.Module):
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
        self.last_encoder = Encoder(input_dim=hidden_dim, hidden_dim=hidden_dim,
                         latent_dim=latent_dim)
        self.config = kwargs
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.stop_token = kwargs['stop_token']
        self.padding_value = kwargs['padding_value']
        self.fc2 = nn.Linear(32, latent_dim)
        
        # Слои для VAE: mu и log_var
        # self.fc_mu = nn.Linear(32, latent_dim)
        # self.fc_logvar = nn.Linear(32, latent_dim)
        
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
        mu = self.fc2(z)
        # log_var = self.fc_logvar(z)
        
        # для соблюдения выхода как у LSTM
        return mu, None, (None, None)
    def load(self, path):
        if path is not None:
            self.load_state_dict(torch.load(path))
class DecoderRNN(nn.Module):
    """
    Декодер на базе LSTM с возможностью предсказания длины и масс-спектра.

    get from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

    Аргументы:
        latent_dim (int): Размерность латентного вектора.
        hidden_size (int): Размер скрытого состояния.
        output_size (int): Размер выходного вектора (совпадает с input_dim энкодера).
        start_token (Tensor): Токен начала последовательности.
        lstm2 (bool): Добавить ли второй LSTM-слой.
        lstm3 (bool): Добавить ли третий LSTM-слой.
        num_part (int): Количество классов в масс-спектре.
    """
    def __init__(self, latent_dim, hidden_size, output_size, start_token: Tensor, 
                lstm2: bool = False, lstm3: bool = False,
                num_part: int = 2,
                reconstruction_params = None,
                **kwargs):
        super(DecoderRNN, self).__init__()
        self.lat2hid = nn.Linear(latent_dim, hidden_size)
        self.lat2hid2 = nn.Linear(latent_dim, hidden_size)
        self.emb_fc = nn.Linear(output_size, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.lrealu = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.max_lenght = 100  # Maximum length of the output sequence
        self.start_token = start_token
        # predict lenght of sequences
        self.fc_seq = nn.Linear(latent_dim, hidden_size)
        self.fc_seq2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.reconstruction_params = reconstruction_params
        # mass spectrum
        self.num_part = num_part
        # if num_part is not None:
        if reconstruction_params is not None:
            self.heads = nn.ModuleList() 
            for i in range(len(reconstruction_params)):
                head = nn.Sequential(
                    nn.Linear(latent_dim, hidden_size, dtype=torch.float32),
                    nn.ELU(),
                    nn.Linear(hidden_size, hidden_size, dtype=torch.float32),
                    nn.ELU(),
                    nn.Linear(hidden_size, 1, dtype=torch.float32),
                )
                self.heads.append(head)

        
        self.lstm2 = None
        self.lstm3 = None
        if lstm2:
            self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        if lstm3:
            self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        self = self.float()
    def forward(self, encoder_hidden, seq_len, x_coordinate:torch.Tensor = None):
        # was forward(self,encoder_outputs, encoder_hidden, seq_len):
        # encoder_outputs - unneeded

        """
        Генерация выходной последовательности, длины и масс-спектра.

        Аргументы:
            encoder_hidden (Tensor): Латентный вектор (обычно mu).
            seq_len (int): Целевая длина выходной последовательности.

        Возвращает:
            Tuple[Tensor, Any, Tensor, Tensor]:
                - Сгенерированная последовательность.
                - Последнее скрытое состояние.
                - Предсказанная длина (число детекторов).
                - Предсказание масс-спектра.
        """
        # start_token shape (1,6)
        batch_size = encoder_hidden.size(0)
        # Подготовка начального вектора
        decoder_input = self.emb_fc(self.start_token).unsqueeze(1)# Zero input for the first timestep
        decoder_input = torch.repeat_interleave(decoder_input, batch_size, dim=0) # Repeat decoder_input for each timestep in the batch
        # Подготовка начальных временых слоев (2 для LSTM)
        # decoder_input mast have shape (batch, 1, hidden_size)
        encoder_hidden1 = self.lat2hid(encoder_hidden)
        encoder_hidden1 = self.lrealu(encoder_hidden1) # ad for 1st exp
        decoder_hidden1 = encoder_hidden1.unsqueeze(0) # Use last hidden state from encoder as initial hidden state for decoder
        # Второй слой 
        encoder_hidden2 = self.lat2hid2(encoder_hidden)
        encoder_hidden2 = self.lrealu(encoder_hidden2) # ad for 1st exp
        decoder_hidden2 = encoder_hidden2.unsqueeze(0) # Use last hidden state from encoder as initial hidden state for decoder
        # Объединение этих слоев
        decoder_hidden =  (decoder_hidden1, decoder_hidden2)
        decoder_outputs = []
        # Генерация данных циклом
        for i in range(seq_len):
            # if not(x_coordinate is None):
            #     decoder_input
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)    
            decoder_input = decoder_output # detach from history as input
            decoder_output = self.out(decoder_output)
            decoder_outputs.append(decoder_output)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        #predict lenght of sequences
        num = self.fc_seq(encoder_hidden)
        num = self.lrealu(num)
        num = self.fc_seq2(num)
        num = self.relu(num)

        return decoder_outputs, None, num, None
    def seq_LSTMs(self, x, hidden = None):
        """
        Последовательное применение до трёх LSTM-слоёв в декодере.

        Аргументы:
            x (Tensor): Входные данные.
            hidden (Tuple[Tensor, Tensor]): Начальное скрытое состояние.

        Возвращает:
            Tuple[Tensor, Tensor]: Выход и скрытое состояние.
        """
        h, c_n = self.lstm1(x, hidden) 
        if self.lstm2 is not None:
            h, c_n = self.lstm2(h, c_n)
        if self.lstm3 is not None:
            h, c_n = self.lstm3(h, c_n)
        return h, c_n

    def forward_step(self, input, hidden):
        """
        Один шаг декодирования.

        Аргументы:
            input (Tensor): Входной токен на текущем шаге.
            hidden (Tuple[Tensor, Tensor]): Скрытое состояние.

        Возвращает:
            Tuple[Tensor, Tuple]: Выход и новое скрытое состояние.
        """
        # output = self.embedding(input)
        output = input
        output = torch.relu(output)
        # output, hidden = self.lstm(output, hidden)
        output, hidden = self.seq_LSTMs(output, hidden)
        return output, hidden
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
class DecoderTransformer(nn.Module):
    # Пока без позиционированных эмбэдингов
    # Нужна ли маска? какая?
    def __init__(self, latent_dim, hidden_size, output_size, start_token: Tensor,
                num_part: int = 2,num_layers:int =2,
                reconstruction_params = None,
                **kwargs):
        super().__init__()
        self.max_lenght = 100
        self.reconstruction_params = reconstruction_params
        self.MaskToken = torch.nn.Parameter(
                                torch.randn(1,1,6))
        self.pos_embedding = PositionalEncoding(hidden_size, max_len=self.max_lenght)
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(d_model=hidden_size,
                                                                nhead = 2,
                                                                dim_feedforward=256,
                                                                dropout=0.1,
                                                                activation='relu',
                                                                layer_norm_eps=1e-05, 
                                                                batch_first=True, 
                                                                norm_first=False, 
                                                                )
        self.TransformerDecoder = nn.TransformerDecoder(self.TransformerDecoderLayer,
                                                        num_layers=num_layers,
                                                        )
        self.start_token = start_token
        self.lat2hid = nn.Linear(latent_dim, hidden_size)
        self.emb_fc = nn.Linear(output_size, hidden_size)

        # predict lenght of sequences
        self.fc_seq = nn.Linear(latent_dim, hidden_size)
        self.fc_seq2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.lrealu = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        # mass spectrum
        self.num_part = num_part
        # if num_part is not None:

        self.coordinat_layer = nn.ModuleList() 
        for i in range(2):
            coordinat = nn.Sequential(
                    nn.Linear(latent_dim, hidden_size),
                    nn.ELU(),
                    nn.Linear(hidden_size, hidden_size*2),
                    nn.ELU(),
                    nn.Linear(hidden_size*2,self.max_lenght),
                )
            self.coordinat_layer.append(coordinat)

        self.out1 = nn.Linear(hidden_size, 32)
        self.out2 = nn.Linear(32, output_size)

        if reconstruction_params is not None:
            self.heads = nn.ModuleList() 
            for i in range(len(reconstruction_params)):
                head = nn.Sequential(
                    nn.Linear(latent_dim, hidden_size, dtype=torch.float32),
                    nn.ELU(),
                    nn.Linear(hidden_size, hidden_size, dtype=torch.float32),
                    nn.ELU(),
                    nn.Linear(hidden_size, 1, dtype=torch.float32),
                )
                self.heads.append(head)

    def forward(self, encoder_hidden, seq_len_real, x_coordinat = None):
        seq_len = 100
        memory = self.lat2hid(encoder_hidden) # batch, hidden_size
        memory = memory.unsqueeze(1) # batch,1,hiden_size

        #predict lenght of sequences
        num = self.fc_seq(encoder_hidden)
        num = self.lrealu(num)
        num = self.fc_seq2(num)
        num = self.relu(num)
        pred_lengths = torch.round(num).long()

        # coordinats
        #predict lenght of sequences
        if x_coordinat is None:
            x = self.coordinat_layer[0](encoder_hidden)
            y = self.coordinat_layer[1](encoder_hidden)
            x=x.unsqueeze(2) # batch, max_lenght, 1
            y=y.unsqueeze(2)
            coordinats = torch.concat((x,y), dim=2) # batch,  max_lenght, 2,
        else:
            coordinats = x_coordinat

        indices = torch.arange(self.max_lenght, device=encoder_hidden.device).unsqueeze(0)
        indices = torch.repeat_interleave(indices, encoder_hidden.size(0), dim=0) # batch, max_lenght
        thresholds = (pred_lengths + 1)
        padding_mask = indices >= thresholds
        #look all
        # padding_mask = torch.zeros_like(padding_mask)

        # memory = torch.repeat_interleave(memory, seq_len, dim=0)
        #first token
        tgt = self.MaskToken
        tgt = self.emb_fc(tgt)
        tgt = torch.repeat_interleave(tgt, encoder_hidden.size(0), dim=0)
        tgt = torch.repeat_interleave(tgt, self.max_lenght, dim=1) # batch, max_lengt, featches
        if x_coordinat is None:
            tgt[:,:, :2] = coordinats # replace coordinat
        else:
            len_ = x_coordinat.size(1)
            tgt[:,:len_, :3] = x_coordinat

        tgt = self.pos_embedding(tgt)
        decoder_outputs = self.TransformerDecoder(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=padding_mask
        )
        # decoder_outputs = tgt
        decoder_outputs = self.out1(decoder_outputs)
        decoder_outputs = self.lrealu(decoder_outputs)
        decoder_outputs = self.out2(decoder_outputs)


        #slise
        decoder_outputs = decoder_outputs[:, : seq_len_real, :]
        if self.reconstruction_params is not None:
            recon_pred = []
            for head in self.heads:
                recon_pred.append(head(encoder_hidden))  # Применяем каждую голову к входу x
            recon_pred = torch.concat(recon_pred, dim=1)
        else:
            recon_pred = None
        return decoder_outputs, (None,None), num, recon_pred
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
        self.last_encoder = Encoder(input_dim=hidden_dim, hidden_dim=hidden_dim,
                         latent_dim=latent_dim)
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
        log_var = self.fc_logvar(z)
        # mask - 1 для воспомогательных 0 - для реальных 
        real_lenght=  mask.size(1) - mask.sum(dim=1) # 
        # для соблюдения выхода как у LSTM
        return mu, log_var, (real_lenght, real_lenght)
    def load(self, path):
        if path is not None:
            self.load_state_dict(torch.load(path))
class DecoderFullyConnected(nn.Module):
    def __init__(self, latent_dim, hidden_size, output_size, coord_dim=3, **kwargs):
        super().__init__()
        self.output_size = output_size
        self.input_dim = latent_dim + coord_dim + 1
        self.latent_to_detector = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, output_size),
                # nn.LeakyReLU(),
                # nn.Linear(hidden_size, hidden_size*2),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size*2, hidden_size*8),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size*8, hidden_size*2),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size*2, hidden_size),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size, output_size)
        )
    def forvard_one_detector(self, z):
        return self.latent_to_detector(z)
    def forward(self, z, seq_len_real, x_coordinat:torch.Tensor = None):
        batch_size = z.size(0)
        # Длина выхода = длина входного тензора (с padding), как у DecoderTransformerSimple (+1 к real_lenght)
        seq_len = x_coordinat.size(1)
        length_feature = seq_len_real.unsqueeze(1).to(z.dtype)
        detector_outputs = torch.zeros((batch_size, seq_len, self.output_size), device=z.device)
        for i in range(seq_len):
            input_inforamtion = torch.cat((z, x_coordinat[:, i, :], length_feature), dim=1)
            detector_outputs[:, i, :] = self.forvard_one_detector(input_inforamtion)
        return detector_outputs, None, seq_len_real.unsqueeze(1), None
class DecoderTransformerSimple(nn.Module):
    def __init__(self, latent_dim, hidden_size, output_size, max_len=100,num_layers=1, **kwargs):
        super().__init__()
        self.max_len = max_len
        # ЗАПРЕЩАЕМ декодеру иметь собственный "стартовый" токен. Всё идёт от z.
        self.latent_to_init = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size * max_len)  # Сразу генерируем все входы
        )
        self.pos_embedding = PositionalEncoding(hidden_size, max_len=max_len)
        
        self.transformer_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=2, dim_feedforward=256, 
            batch_first=True, activation='relu'
        )
        self.transformer = nn.TransformerDecoder(self.transformer_layer, num_layers=num_layers)
        
        # Memory - это преобразованный z. Он статичен для всей последовательности.
        self.latent_to_memory = nn.Linear(latent_dim, hidden_size)
        
        # Выходной слой
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.LeakyReLU(),
            nn.Linear(32, output_size)
        )

        self.lenght_layer = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
        )

    def forward(self, z, seq_len_real, x_coordinat:torch.Tensor = None):
        batch_size = z.size(0)
        memory = self.latent_to_memory(z).unsqueeze(1)  # [batch, 1, hidden]
        
        decoder_input = self.latent_to_init(z)  # [batch, hidden * max_len]
        decoder_input = decoder_input.view(batch_size, self.max_len, -1)  # [batch, max_len, hidden]
        decoder_input = self.pos_embedding(decoder_input)
        if x_coordinat is not None:
            # Заменяем координаты на реальные, если они предоставлены
            coord_len = x_coordinat.size(1)
            decoder_input[:, :coord_len, :3] = x_coordinat[:, :, :3]  # replace coordinat
        #Создаем маску НА ОСНОВЕ ИСТИННОЙ ДЛИНЫ (seq_len_real), а не предсказанной.
        # Это заставляет учиться информацию о длине содержать в z.
        padding_mask = self._generate_padding_mask(batch_size, seq_len_real, z.device)
        
        
        # Убираем tgt_mask для simplicity, оставляем только padding mask.
        output = self.transformer(
            tgt=decoder_input, 
            memory=memory,
            tgt_key_padding_mask=padding_mask
        )
        
        # 5. Применяем выходной слой и обрезаем по реальной длине батча
        output = self.output_layer(output)
        pred_lenght = self.lenght_layer(z)
        # Правильно обрабатываем seq_len_real - это может быть тензор или число
        if isinstance(seq_len_real, torch.Tensor):
            max_len_in_batch = torch.max(seq_len_real).item() + 1
        else:
            max_len_in_batch = int(seq_len_real) + 1
        max_len_in_batch = min(max_len_in_batch, self.max_len)  # Не превышаем max_len
        return output[:, :max_len_in_batch, :], None, pred_lenght, None  # Возвращаем в формате вашего интерфейса

    def _generate_padding_mask(self, batch_size, lengths, device):
        # lengths: [batch] - истинные длины последовательностей в батче
        mask = torch.zeros(batch_size, self.max_len, device=device, dtype=torch.bool)
        # Обрабатываем lengths как тензор или список
        if isinstance(lengths, torch.Tensor):
            lengths_list = lengths.cpu().tolist()
        else:
            lengths_list = lengths if isinstance(lengths, list) else [lengths] * batch_size
        
        for i, length in enumerate(lengths_list):
            length_int = int(length)
            if length_int < self.max_len:
                mask[i, length_int:] = True
        return mask

class VAE(nn.Module):
    """
    Вариационный автокодировщик (VAE), основанный на LSTM энкодере и декодере.

    Аргументы:
        input_dim (int): Размерность входного вектора.
        hidden_dim (int): Размерность скрытого слоя.
        latent_dim (int): Размерность латентного пространства.
        start_token (Tensor): Начальный токен последовательности.
        lstm2 (bool): Использовать ли второй LSTM-слой.
        lstm3 (bool): Использовать ли третий LSTM-слой.
        num_part (int): Количество классов в масс-спектре.
    """
    def __init__(self, input_dim=6, hidden_dim=32, latent_dim=16, start_token: Tensor = torch.zeros(1,6), lstm2: bool = False, lstm3: bool = False, num_part: int = 2,
                 num_layers = 2, hidden_dim_decoder = 128,
                 reconstruction_params=None,
                 CLS_token: Optional[str] = None,
                 reparameterize_koef: float = 1.0,
                 num_layers_decoder:int = 1,
                 denoise_koef: float = 0.0,
                 **kwargs ) -> None:
        super(VAE, self).__init__()
        self.denoise_koef = denoise_koef
        self.TRANSFORMER = True
        self.reparameterize_koef = reparameterize_koef
        self.padding_value = kwargs.get('padding_value', None)
        if CLS_token == 'trainable':
            self.CLS_token = torch.nn.Parameter(
                                start_token.unsqueeze(0))
        else:
            self.CLS_token = None
        if self.TRANSFORMER:

        # self.encoder = Encoder(input_dim, hidden_dim, latent_dim, lstm2=lstm2, lstm3=lstm3 )
            self.encoder = Encoder_Transformer(input_dim, hidden_dim, latent_dim,
                                                num_layers=num_layers, **kwargs)
            self.decoder = DecoderFullyConnected(latent_dim, hidden_dim_decoder, input_dim)
            # self.decoder = DecoderTransformerSimple(latent_dim, hidden_dim_decoder, input_dim,
            #                                         max_len = 100,
            #                                         num_layers=num_layers_decoder)
        else:
            self.encoder = Encoder(input_dim, hidden_dim, latent_dim, lstm2=lstm2, lstm3=lstm3 )
            self.decoder = DecoderRNN(latent_dim, hidden_dim_decoder, input_dim, start_token,
                                    lstm2=lstm2, lstm3=lstm3, num_part=num_part)
        # self.decoder = DecoderTransformer(latent_dim, hidden_dim, input_dim, start_token, num_part=num_part)
        print("Encoder has params:", self.count_parameters(self.encoder),"Decoder has params:", self.count_parameters(self.decoder))
    def reparameterize(self, mu, log_var, koef=1):
        """
        Реализация трика репараметризации.

        Аргументы:
            mu (Tensor): Средние латентного распределения.
            log_var (Tensor): Лог-дисперсии.
            koef (float): Коэффициент масштабирования (например, для beta-VAE).

        Возвращает:
            Tensor: Выборка из латентного пространства.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std * koef
    def add_noise(self, x):
        signal = x[:,:,3:4]
        flat = x[:,:,4:5]
        real = x[:,:,5:6]
        
        signal_add = torch.abs(torch.min(signal, dim=1, keepdim=True)[0]) # (batch, 1, 1)
        flat_add = torch.mean(torch.abs(torch.diff(flat, dim=1)), dim=1, keepdim=True)[0] # (batch, 1, 1)
        real_add = torch.abs(torch.min(real, dim=1, keepdim=True)[0]) # (batch, 1, 1)

        noise_signal = self.denoise_koef * torch.randn_like(signal)# * torch.sqrt(signal_add)
        noise_flat = self.denoise_koef * torch.randn_like(flat)# * torch.sqrt(flat_add)
        noise_real = self.denoise_koef * torch.randn_like(real)# * torch.sqrt(real_add)

        x[:,:,3:4] += noise_signal
        x[:,:,4:5] += noise_flat
        x[:,:,5:6] += noise_real
        return x
    def forward(self, x):
        seq_len = x.size(1)
        # change start token on CLS token

        if self.CLS_token is not None:
            x[:,0,:] = torch.repeat_interleave(self.CLS_token, x.size(0), dim=0)

        # Вычисляем реальные длины последовательностей (исключая padding)
        # padding_value обычно -11.0, проверяем по первому признаку
        #train mode
        if self.training:
            x = self.add_noise(x)
        if hasattr(self, 'padding_value') and self.padding_value is not None:
            padding_mask = (x[:, :, 0] != self.padding_value)  # (batch, seq_len)
            lengths = padding_mask.sum(dim=1)  # (batch,)
        else:
            lengths = None
        
        # Используем z (репараметризованный вектор) вместо mu
        if self.TRANSFORMER:

            mu, log_var, (real_lenght, _) = self.encoder(x)

            # Репараметризация для VAE
            z = self.reparameterize(mu, log_var, koef=self.reparameterize_koef)
            recon_x, _, num, mass = self.decoder(z, real_lenght, x[:,:,:3])
        else:
            mu, log_var, (real_lenght, _) = self.encoder(x, lengths=lengths)

            # Репараметризация для VAE
            z = self.reparameterize(mu, log_var, koef=self.reparameterize_koef)
            recon_x, _, num, mass = self.decoder(z, seq_len)
        return recon_x, mu, log_var, num, mass
    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    def load(self, path):
        if path is not None:
            strict = True
            print(f"load {strict} format. If it is 'FALSE' we worning" )
            self.load_state_dict(torch.load(path), strict=strict)

if __name__ == "__main__":
    model = VAE(stop_token=-11, padding_value=-10, num_layers_decoder = 1)
    x=torch.randn(2,10,6)
    print('input shape:', x.shape)
    res = model(x)
    print('recon shape:', res[0].shape)
    print('mu shape:', res[1].shape)
    print('log_var shape:', res[2].shape)
    print('num shape:', res[3].shape)
    print('mass shape:', res[4].shape)