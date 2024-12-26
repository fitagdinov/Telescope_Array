from torch import nn
import torch 
from torch import Tensor
# from logging import logging
# loger = logging.getLogger(__name__)

class Embading(nn.Module):
    def __init__(self):
        super(Embading, self).__init__()



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim*2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim*2, latent_dim)
    def forward(self, x):
        h, c_n = self.lstm(x)  # h_n shape: (1, batch_size, hidden_dim)
        # for LSTM
        c_n = torch.concat((c_n[0], c_n[1]), dim=2)
        h_n = c_n.squeeze(0)  # убираем первую размерность
        mu = self.fc_mu(h_n)  # среднее латентного пространства
        log_var = self.fc_logvar(h_n)  # логарифм дисперсии латентного пространства
        return mu, log_var, (h_n, c_n)
    

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    def forward(self, z, seq_len):
        h = torch.relu(self.fc(z)).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        h = h.repeat(1, seq_len, 1)  # Повторяем скрытое состояние для каждого шага времени
        lstm_out, _ = self.lstm(h)  # Проходим через LSTM
        
        return self.output_layer(lstm_out)
    
class DecoderRNN(nn.Module):
    '''
    get from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    '''
    def __init__(self, latent_dim, hidden_size, output_size, start_token: Tensor):
        super(DecoderRNN, self).__init__()
        self.lat2hid = nn.Linear(latent_dim, hidden_size)
        self.lat2hid2 = nn.Linear(latent_dim, hidden_size)
        self.emb_fc = nn.Linear(output_size, hidden_size)
        # self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.lrealu = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.max_lenght = 100  # Maximum length of the output sequence
        self.start_token = start_token
        # predict lenght of sequences
        self.fc_seq = nn.Linear(latent_dim, hidden_size)
        self.fc_seq2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
    def forward(self, encoder_outputs, encoder_hidden, seq_len):
        # start_token shape (1,6)
        batch_size = encoder_outputs.size(0)
        decoder_input = self.emb_fc(self.start_token).unsqueeze(1)# Zero input for the first timestep
        decoder_input = torch.repeat_interleave(decoder_input, batch_size, dim=0) # Repeat decoder_input for each timestep in the batch
        # decoder_input mast have shape (batch, 1, hidden_size)
        encoder_hidden1 = self.lat2hid(encoder_hidden)
        encoder_hidden1 = self.lrealu(encoder_hidden1) # ad for 1st exp
        decoder_hidden1 = encoder_hidden1.unsqueeze(0) # Use last hidden state from encoder as initial hidden state for decoder
        # for LSTM
        encoder_hidden2 = self.lat2hid2(encoder_hidden)
        encoder_hidden2 = self.lrealu(encoder_hidden2) # ad for 1st exp
        decoder_hidden2 = encoder_hidden2.unsqueeze(0) # Use last hidden state from encoder as initial hidden state for decoder
        decoder_hidden =  (decoder_hidden1, decoder_hidden2)
        decoder_outputs = []
        for i in range(seq_len):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)    
            decoder_input = decoder_output # detach from history as input
            decoder_output = self.out(decoder_output)
            decoder_outputs.append(decoder_output)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        #predict lenght of sequences
        num = self.fc_seq(encoder_outputs)
        num = self.lrealu(num)
        num = self.fc_seq2(num)
        num = self.relu(num)
        return decoder_outputs, decoder_hidden, num

    def forward_step(self, input, hidden):
        # output = self.embedding(input)
        output = input
        output = torch.relu(output)
        output, hidden = self.lstm(output, hidden)
        return output, hidden

class VAE(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, latent_dim=1, start_token: Tensor = torch.zeros(1,6)) -> None:
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        # self.decoder = Decoder(latent_dim, hidden_dim, input_dim, hidden_dim_latent)
        self.decoder = DecoderRNN(latent_dim, hidden_dim, input_dim, start_token)
        print("Encoder has params:", self.count_parameters(self.encoder),"Decoder has params:", self.count_parameters(self.decoder))
    def reparameterize(self, mu, log_var, koef=1):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std * koef
    def forward(self, x):
        seq_len = x.size(1)
        mu, log_var, (h_n, c_n) = self.encoder(x)
        z = self.reparameterize(mu, log_var, koef=1.0)
        #change second param
        recon_x, _, num = self.decoder(z, mu, seq_len)
        return recon_x, mu, log_var, num
    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)