from torch import nn
import torch 
from torch import Tensor
# from logging import logging
# loger = logging.getLogger(__name__)

class Encoder(nn.Module):
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
        h, c_n = self.lstm1(x) 
        if self.lstm2 is not None:
            h, c_n = self.lstm2(h)
        if self.lstm3 is not None:
            h, c_n = self.lstm3(h)
        return h, c_n

    def forward(self, x):
        h, c_n = self.seq_LSTMs(x)  # h_n shape: (1, batch_size, hidden_dim)
        # for LSTM
        c_n = torch.concat((c_n[0], c_n[1]), dim=2)
        h_n = c_n.squeeze(0)  # убираем первую размерность
        mu = self.fc_mu(h_n)  # среднее латентного пространства
        log_var = self.fc_logvar(h_n)  # логарифм дисперсии латентного пространства
        return mu, log_var, (h_n, c_n)
    
class DecoderRNN(nn.Module):
    '''
    get from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    '''
    def __init__(self, latent_dim, hidden_size, output_size, start_token: Tensor, lstm2: bool = False, lstm3: bool = False, num_part: int = 2):
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

        # predict mass spectrum
        mass = self.fc_mass(encoder_outputs)
        mass = self.lrealu(mass)
        mass = self.fc_mass2(mass)
        mass = self.lrealu(mass)
        mass = self.fc_mass3(mass)
        mass = self.sofrmax(mass)
        return decoder_outputs, decoder_hidden, num, mass
    def seq_LSTMs(self, x, hidden = None):
        h, c_n = self.lstm1(x, hidden) 
        if self.lstm2 is not None:
            h, c_n = self.lstm2(h, c_n)
        if self.lstm3 is not None:
            h, c_n = self.lstm3(h, c_n)
        return h, c_n

    def forward_step(self, input, hidden):
        # output = self.embedding(input)
        output = input
        output = torch.relu(output)
        # output, hidden = self.lstm(output, hidden)
        output, hidden = self.seq_LSTMs(output, hidden)
        return output, hidden

class VAE(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, latent_dim=1, start_token: Tensor = torch.zeros(1,6), lstm2: bool = False, lstm3: bool = False, num_part: int = 2) -> None:
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, lstm2=lstm2, lstm3=lstm3 )
        # self.decoder = Decoder(latent_dim, hidden_dim, input_dim, hidden_dim_latent)
        self.decoder = DecoderRNN(latent_dim, hidden_dim, input_dim, start_token, lstm2=lstm2, lstm3=lstm3, num_part=num_part)
        print("Encoder has params:", self.count_parameters(self.encoder),"Decoder has params:", self.count_parameters(self.decoder))
    def reparameterize(self, mu, log_var, koef=1):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std * koef
    def forward(self, x):
        seq_len = x.size(1)
        mu, log_var, (h_n, c_n) = self.encoder(x)
        z = self.reparameterize(mu, log_var, koef=1.0)
        
        # change 05.03.2025. Was self.decoder(z, mu, seq_len)
        recon_x, _, num, mass = self.decoder(mu, mu, seq_len)
        return recon_x, mu, log_var, num, mass
    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    def load(self, path):
        if path is not None:
            self.load_state_dict(torch.load(path))
        