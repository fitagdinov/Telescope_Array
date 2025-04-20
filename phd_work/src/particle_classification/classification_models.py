from torch import nn
import torch 
from torch import Tensor
from typing import Optional
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../train_VAE')))

from model import Encoder
class Simple_classifiacation_model(nn.Module):
    def __init__(self, encoder_path: Optional[str] = None, num_class: int = 2, **kwargs):
        super().__init__()
        self.encoder = Encoder(input_dim=kwargs['input_dim'], hidden_dim=kwargs['hidden_dim'], latent_dim=kwargs['latent_dim'])
        self.encoder_path = encoder_path
        if encoder_path is not None:
            load_dict =  torch.load(encoder_path)
            load_dict_endoder = {}
            for k, v in load_dict.items():
                if 'encoder' in k:
                    load_dict_endoder[k.replace('encoder.','')] = v
            del(load_dict)
            self.encoder.load_state_dict(load_dict_endoder)
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.latent_class = 16
        self.fc1 = nn.Linear(kwargs['latent_dim'], self.latent_class)
        self.fc2 = nn.Linear(self.latent_class, num_class)
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.LeakyReLU()
    def forward(self, x):
        mu, log_var, _ = self.encoder(x)
        z = self.fc1(mu)
        z = self.activation(z)
        z = self.fc2(z)
        # z = self.softmax(z)
        return z
    def load(self, path):
        if path is not None:
            self.load_state_dict(torch.load(path))
class TransformerClassificationModel(nn.Module):
    def __init__(self, encoder_path: Optional[str] = None, num_class: int = 2, **kwargs):
        super().__init__()
        # TODO
        """
        enbading fc 6*64

        4 layres
        dm = 64
        dff = 256
        head = 4
        """
        self.embading  = nn.Linear(6,64)
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=64,
                                                                nhead = 4,
                                                                dim_feedforward=256,
                                                                dropout=0.1,
                                                                activation='relu',
                                                                layer_norm_eps=1e-05, 
                                                                batch_first=True, 
                                                                norm_first=False, 
                                                                )
        self.TransformerEncoder = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=4,
                                                        )
        self.config = kwargs
        self.latent_class = 16
        self.fc1 = nn.Linear(64, self.latent_class)
        self.fc2 = nn.Linear(self.latent_class, num_class)
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
    def get_mask(self, x, stop_token = None, padding_value = None):
        if stop_token is None:
            stop_token = torch.tensor(self.config['stop_token'], dtype=torch.long, device=x.device)
        if padding_value is None:
            padding_value = torch.tensor(self.config['padding_value'], dtype=torch.long, device=x.device)
        mask = torch.zeros_like(x, dtype=torch.long)  # Убедитесь, что это long (int64)
        mask = torch.where(x == stop_token, torch.tensor(1, dtype=torch.long, device=x.device), mask)
        mask = torch.where(x == padding_value, torch.tensor(1, dtype=torch.long, device=x.device), mask)
        mask[:,0,:] = 1
        mask = mask[:,:,0] # need (batch, seq)
        return mask.bool().to(x.device)


    def forward(self,x):
        mask = self.get_mask(x)
        
        x = self.embading(x)
        x = self.TransformerEncoder(x, src_key_padding_mask= mask)
        x = torch.mean(x, dim=1)
        z = self.fc1(x)
        z = self.activation(z)
        z = self.fc2(z)
        x = self.softmax(x)
        return x
    def load(self, path):
        if path is not None:
            self.load_state_dict(torch.load(path))
if __name__ == '__main__':
    model = Simple_classifiacation_model(encoder_path="/home/rfit/Telescope_Array/phd_work/src/train_VAE/test_particles/Pr_Phootn_latent_64latent_dim=64;_paticles=['pr', 'photon'];_/best",
                                                               input_dim=6, hidden_dim=512, latent_dim=64)
    
    optimizer = torch.optim.SGD(model.parameters(), lr= 0.1)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(64, 64, 6)
    y = torch.randint(0, 2, (64,))
    for i in range(10):
        optimizer.zero_grad()
        y_pred, mu = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        print('y_pred\n', y_pred,'\nencoder\n', mu)


