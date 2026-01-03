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
print(os.getenv('CUDA_VISIBAL_DEVICE'))
MODE = "ITERATE" #["MEAN", "ITERATE"]
class BertAugmentation():
    def __init__(self,
        embading_path:str = '/home/rfit/Telescope_Array/phd_work/Models/AutoEncoder/Proton_train_lat=16/best',
        bert_path:str = '/home/rfit/Telescope_Array/phd_work/Models/BERT/BERT_one_work/best'):
        self.pipline_BERT = PiplineMask('phd_work/src/BERT/config.yaml')
        self.pipline_BERT.model.load(bert_path)
        VAE_state_dict = torch.load(embading_path)
        self.encoder_model = Encoder_Transformer(6,128, 16,
                            stop_token = self.pipline_BERT.config['stop_token'],
                            padding_value=self.pipline_BERT.config['padding_value'])
        embading_state_dict = {}
        for k,v in VAE_state_dict.items():
            if 'encoder' in k:
                newk = k.replace('encoder.', '')
                # for transformer
                if 'last_' in k:
                    newk = newk.replace('last_', 'last_encoder.')
                embading_state_dict[newk] = v
        print(embading_state_dict.keys())
        print(self.encoder_model.state_dict().keys())
        self.encoder_model.load_state_dict(embading_state_dict)
        self.encoder_model.to('cuda:0')
        self.encoder_model.eval()
        self.particles = ['proton', 'photon']
    def run(self,nums:int=1, dl_n:int=0,
            save_dir:str = '/home/rfit/Telescope_Array/phd_work/src/BERT/augmentation_research/runs/proton'):
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        columns = ['id', 'particle', 'emb:start']
        columns += [f'emb:{i+1}' for i in range(nums)]
        df = pd.DataFrame(columns = columns)
        self.pipline_BERT.model.eval()
        embads_dict = {}
        pbar = tqdm(self.pipline_BERT.val_loaders[dl_n])
        n=0
        particle = self.particles[dl_n]
        for x, _, _ in pbar:
            x = x.to('cuda:0')
            emb,_,_ = self.encoder_model(x)
            if 0 in embads_dict.keys():
                embads_dict[0] = np.concatenate((embads_dict[0], 
                        emb.to('cpu').detach().numpy()), 
                        axis = 0)
            else:
                embads_dict[0] = emb.to('cpu').detach().numpy()
            if MODE == "ITERATE":
                for i in range(1, nums+1):
                    x = self.pipline_BERT.run_ones(x)
                    emb,_,_ = self.encoder_model(x)
                    if i in embads_dict.keys():
                        embads_dict[i] = np.concatenate((embads_dict[i], 
                                emb.to('cpu').detach().numpy()), 
                                axis = 0)
                    else:
                        embads_dict[i] = emb.to('cpu').detach().numpy()
            elif MODE == "MEAN":
                mean_emb = np.zeros_like(emb.to('cpu').detach().numpy())
                for i in range(1, nums+1):
                    x = self.pipline_BERT.run_ones(x)
                    emb,_,_ = self.encoder_model(x)
                    mean_emb += emb.to('cpu').detach().numpy()
                mean_emb = mean_emb/nums
                if "mean" in embads_dict.keys():
                    embads_dict['mean'] = np.concatenate((embads_dict['mean'], 
                            mean_emb), 
                            axis = 0)
                else:
                    embads_dict['mean'] = mean_emb
        #saving
        for k,v in embads_dict.items():
            file = os.path.join(save_dir, f'embading_{k}.npy')
            np.save(file, v)
        print("SAVING IS FINISH")
        
# embading_path = '/home/rfit/Telescope_Array/phd_work/Models/AutoEncoder/Encoder_CLS FIRST_CONT_DecoderTransformer_LR305/last'
embading_path = '/home/rfit/Telescope_Array/phd_work/Models/AutoEncoder/small_decoder/last'
embading_path = '/home/rfit/Telescope_Array/phd_work/Models/AutoEncoder/variavle_KL/best'
run_dir = '/home/rfit/Telescope_Array/phd_work/src/BERT/augmentation_research/variavle_KL' + MODE
os.makedirs(run_dir, exist_ok=True)
augmentation_pipline = BertAugmentation(embading_path = embading_path)
augmentation_pipline.run(nums=8, dl_n = 0, save_dir = os.path.join(run_dir, 'proton'))

augmentation_pipline.run(nums=8, dl_n = 1,save_dir = os.path.join(run_dir, 'photon'))