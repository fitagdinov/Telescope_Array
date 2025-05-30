from torch import nn
import torch 
from torch import Tensor
from typing import Optional
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
    def seq_LSTMs(self, x):
        """
        Последовательно применяет до трёх LSTM-слоёв.

        Аргументы:
            x (Tensor): Последовательность входных данных.

        Возвращает:
            Tuple[Tensor, Tensor]: Последний выход и скрытое состояние.
        """
        h, c_n = self.lstm1(x) 
        if self.lstm2 is not None:
            h, c_n = self.lstm2(h)
        if self.lstm3 is not None:
            h, c_n = self.lstm3(h)
        return h, c_n

    def forward(self, x):
        """
        Прямой проход энкодера.

        Аргументы:
            x (Tensor): Входная последовательность формы (batch, seq_len, input_dim).

        Возвращает:
            mu (Tensor): Вектор средних латентного распределения.
            log_var (Tensor): Логарифм дисперсий латентного распределения.
            Tuple[Tensor, Tensor]: Последние состояния h_n и c_n.
        """
        h, c_n = self.seq_LSTMs(x)  # h_n shape: (1, batch_size, hidden_dim)
        # for LSTM
        c_n = torch.concat((c_n[0], c_n[1]), dim=2)
        h_n = c_n.squeeze(0)  # убираем первую размерность
        mu = self.fc_mu(h_n)  # среднее латентного пространства
        # Для АЕ не надо, но для VAE пусть останется
        log_var = self.fc_logvar(h_n)  # логарифм дисперсии латентного пространства
        # (h_n, c_n) - различаются только shape
        return mu, log_var, (h_n, c_n)
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
                                                                dim_feedforward=128,
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
        # self.last_encoder = Encoder(input_dim=hidden_dim, hidden_dim=hidden_dim,
        #                  latent_dim=latent_dim)
        self.config = kwargs
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.stop_token = kwargs['stop_token']
        self.padding_value = kwargs['padding_value']
        self.fc2 = nn.Linear(32, latent_dim)
        
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

        mask[:,0,:] = 1
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


        z = self.fc1(CLS)
        z = self.activation(z)
        z = self.fc2(z)
        # z = self.activation(z)
        # x = self.softmax(x)
        
        # для соблюдения выхода как у LSTM
        return z, None, (None, None)
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
    def __init__(self, latent_dim, hidden_size, output_size, start_token: Tensor, lstm2: bool = False, lstm3: bool = False, num_part: int = 2, **kwargs):
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

        # mass spectrum
        self.num_part = num_part
        # if num_part is not None:
        self.fc_mass = nn.Linear(latent_dim, hidden_size)
        self.fc_mass2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mass3 = nn.Linear(hidden_size, num_part)
        self.sofrmax = nn.Softmax()

        
        self.lstm2 = None
        self.lstm3 = None
        if lstm2:
            self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        if lstm3:
            self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
    def forward(self, encoder_hidden, seq_len):
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

        # predict mass spectrum
        mass = self.fc_mass(encoder_hidden)
        mass = self.lrealu(mass)
        mass = self.fc_mass2(mass)
        mass = self.lrealu(mass)
        mass = self.fc_mass3(mass)
        mass = self.sofrmax(mass)
        return decoder_outputs, decoder_hidden, num, mass
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

class DecoderTransformer(nn.Module):
    # Пока без позиционированных эмбэдингов
    # Нужна ли маска? какая?
    def __init__(self, latent_dim, hidden_size, output_size, start_token: Tensor,
                  num_part: int = 2,num_layers:int =2, **kwargs):
        super().__init__()
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(d_model=hidden_size,
                                                                nhead = 4,
                                                                dim_feedforward=64,
                                                                dropout=0.1,
                                                                activation='relu',
                                                                layer_norm_eps=1e-05, 
                                                                batch_first=True, 
                                                                norm_first=False, 
                                                                )
        self.TransformerDecoder = nn.TransformerDecoder(self.TransformerDecoderLayer, num_layers=num_layers,
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
        self.fc_mass = nn.Linear(latent_dim, hidden_size)
        self.fc_mass2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mass3 = nn.Linear(hidden_size, num_part)
        self.sofrmax = nn.Softmax()

        self.out1 = nn.Linear(hidden_size, 32)
        self.out2 = nn.Linear(32, output_size)
    def forward(self, encoder_hidden, seq_len):
        memory = self.lat2hid(encoder_hidden) # batch, hidden_size
        memory = memory.unsqueeze(1) # batch,1,hiden_size
        # memory = torch.repeat_interleave(memory, seq_len, dim=0)
        #first token
        tgt = self.start_token.unsqueeze(1)
        tgt = self.emb_fc(tgt)
        tgt = torch.repeat_interleave(tgt, encoder_hidden.size(0), dim=0)
        decoder_outputs = []
        for i in range(seq_len-1):
            # memory update? probably NOT
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to('cuda')
            
            # print(tgt.shape, memory.shape, tgt_mask.shape)
            output = self.TransformerDecoder(
                tgt=tgt,  # (batch, seq_len, d_model)
                memory=memory[:,:], # tgt.size(1) ???
                tgt_mask=tgt_mask
            )
            token = output[:,-1, :].unsqueeze(1)
            # print('output', output.shape, token.shape, tgt.shape)
            tgt = torch.concat((tgt, token), dim=1)

        decoder_outputs = tgt
        decoder_outputs = self.out1(decoder_outputs)
        decoder_outputs = self.lrealu(decoder_outputs)
        decoder_outputs = self.out2(decoder_outputs)
        #predict lenght of sequences
        num = self.fc_seq(encoder_hidden)
        num = self.lrealu(num)
        num = self.fc_seq2(num)
        num = self.relu(num)

        # predict mass spectrum
        mass = self.fc_mass(encoder_hidden)
        mass = self.lrealu(mass)
        mass = self.fc_mass2(mass)
        mass = self.lrealu(mass)
        mass = self.fc_mass3(mass)
        mass = self.sofrmax(mass)
        
        return decoder_outputs, (None,None), num, mass

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
    def __init__(self, input_dim=5, hidden_dim=32, latent_dim=1, start_token: Tensor = torch.zeros(1,6), lstm2: bool = False, lstm3: bool = False, num_part: int = 2,
                 num_layers = 2,
                 **kwargs ) -> None:
        super(VAE, self).__init__()
        # self.encoder = Encoder(input_dim, hidden_dim, latent_dim, lstm2=lstm2, lstm3=lstm3 )
        self.encoder = Encoder_Transformer(input_dim, hidden_dim, latent_dim,
                                            num_layers=num_layers, **kwargs)
        # self.decoder = Decoder(latent_dim, hidden_dim, input_dim, hidden_dim_latent)
        hidden_dim_decoder = 512
        self.decoder = DecoderRNN(latent_dim, hidden_dim_decoder, input_dim, start_token, lstm2=lstm2, lstm3=lstm3, num_part=num_part)
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
    def forward(self, x):
        seq_len = x.size(1)
        mu, log_var, (h_n, c_n) = self.encoder(x)

        # Unneeded, but that for remember 
        # z = self.reparameterize(mu, log_var, koef=1.0)
        
        # change 05.03.2025. Was self.decoder(z, mu, seq_len)
        recon_x, _, num, mass = self.decoder(mu, seq_len)
        return recon_x, mu, log_var, num, mass
    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    def load(self, path):
        if path is not None:
            self.load_state_dict(torch.load(path))


if __name__ == "__main__":
    decoder = DecoderTransformer(16,128,6,start_token=torch.zeros(1,6))
    latent = torch.randn(10,16).to('cpu')
    res = decoder(latent, 3)
    print([r.shape for r in res[0:1]+res[2:]])

    decoder = DecoderRNN(16,128,6,start_token=torch.zeros(1,6))
    latent = torch.randn(10,16).to('cpu')
    res = decoder(latent, 3)
    print([r.shape for r in res[0:1]+res[2:]])

        