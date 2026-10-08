import sys
sys.path.append('phd_work')
sys.path.append('phd_work/src/BERT')
import pandas as pd
from src.BERT.pipline import PiplineMask
import torch
from src.train_VAE.model import Encoder, Encoder_Transformer_AE, Encoder_Transformer
import os
import numpy as np
from tqdm import tqdm
import shutil 
import random
print(os.getenv('CUDA_VISIBLE_DEVICES'))

import matplotlib.pyplot as plt
def show_data(data, predict, mask, fig_axs = None, lenght =20):
    names = ['det x', 'det y', 'det z', 'signal', 'flat front', '(real - front)']    
    if fig_axs is None:
        fig, axs = plt.subplots(2,3, figsize = (30,30))
    else:
        fig, axs = fig_axs
    legend = []
    legend.append(f'fake')
    legend.append('true')
    legend.append('mask')
    for i in range(6):
        row = i%2
        col = i//2

        axs[row][col].plot(predict.to('cpu').detach().numpy()[1:lenght,i],)
        axs[row][col].plot(data.to('cpu').detach().numpy()[1:lenght,i], 'b+-')
        axs[row][col].plot(mask.to('cpu').detach().numpy()[1:lenght,i], 'go')
        axs[row][col].set_title(f'chanal {names[i]}')
        axs[row][col].legend(legend)
    return fig
    
class BertAugmentation():
    def __init__(self,
        embading_path:str = '/home/rfit/Telescope_Array/phd_work/Models/AutoEncoder/Proton_train_lat=16/best',
        bert_path:str = '/home/rfit/Telescope_Array/phd_work/Models/BERT/BERT_one_work/best'):
        self.pipline_BERT = PiplineMask('phd_work/src/BERT/config.yaml')
        self.pipline_BERT.model.load(bert_path)
        self.particles = ['proton', 'photon']
    def run(self,nums:int=1, dl_n:int=0):
        self.pipline_BERT.model.eval()
        pbar = tqdm(self.pipline_BERT.val_loaders[dl_n])
        for x, _, _ in pbar:
            x = x.to('cuda:0')
            original_data = x.clone()
            x, x_mask = self.pipline_BERT.run_ones(x)
            x2, x_mask2 = self.pipline_BERT.run_ones_without_remask(x)
            break
        print(original_data.shape, x.shape, x_mask.shape)
        random_index = random.choices(list(np.arange(x.shape[0])), k=3)
        for j,i in enumerate(random_index):
            original_data_ = original_data[i]
            x_ = x[i]
            x_mask_ = x_mask[i]
            fig = show_data(original_data_,x_, x_mask_ )
            plt.savefig(f'phd_work/src/BERT/examples/example_BERT_remask_{str(j)}.png')
            
            x_ = x2[i]
            x_mask_ = x_mask2[i]
            fig = show_data(original_data_,x_, x_mask_ )
            plt.savefig(f'phd_work/src/BERT/examples/example_BERT_UNremask_{str(j)}.png')
        
# embading_path = '/home/rfit/Telescope_Array/phd_work/Models/AutoEncoder/Encoder_CLS FIRST_CONT_DecoderTransformer_LR305/last'

augmentation_pipline = BertAugmentation(embading_path = None)
augmentation_pipline.run(nums=8, dl_n = 0)