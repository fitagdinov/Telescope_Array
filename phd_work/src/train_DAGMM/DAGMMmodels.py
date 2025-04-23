from torch import nn
import torch 
from torch import Tensor
from typing import Optional
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../train_VAE')))
from model import Encoder, DecoderRNN, VAE

class Estimation(nn.Module):
    def __init__(self, output_dim, input_dim,  **kwargs):
        super(Estimation).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        # NN
        lat1=64
        lat2 = 32 
        self.fc1 = nn.Linear(input_dim, lat1)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(lat1, lat2)
        self.fc3 = nn.Linear(lat2, output_dim)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x=self.fc1(x)
        x=self.activation(x)
        x=self.fc2(x)
        x=self.activation(x)
        x=self.fc3(x)
        x=self.softmax(x)
        return x
