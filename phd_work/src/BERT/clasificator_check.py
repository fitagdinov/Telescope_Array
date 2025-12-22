import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../train_VAE')))
device = 'cuda'
from particle_classification.classification_models import TransformerClassificationModel, Simple_classifiacation_model
from particle_classification.classification_metrics import ClassificationMetrics
from train_VAE.utils import get_time, get_params_str, show_pred, read_config, clean_mask, MCPAR_index2srt
import model as Model
import train_VAE.datasets as DataSet
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
from typing import List, Optional
from torch.utils.tensorboard import SummaryWriter
import numpy as np
def init_class_model():
    class_conf = read_config('/home/rfit/Telescope_Array/phd_work/src/particle_classification/classification_config.yaml')
    classificator = TransformerClassificationModel(**class_conf)
    class_path = '/home/rfit/Telescope_Array/phd_work/Models/Classification/test_particles/One_working_V2/best'
    classificator.load(class_path)
    classificator.eval()
    return classificator.to(device)
def init_BERT():
    config = read_config('/home/rfit/Telescope_Array/phd_work/src/BERT/config.yaml')
    print(config)
    mask_model = Model.EncoderTransformerMask(input_dim=config['input_dim'],
                                hidden_dim=config['hidden_dim'], 
                                latent_dim=config['latent_dim'],
                                stop_token=config['stop_token'],
                                num_layers=config['num_layers'],
                                padding_value=config['padding_value'],
                                start_token=config['start_token'],

                                used_MAE = config['used_MAE']
                                ).to(device)
                        
    # model_path = '/home/rfit/Telescope_Array/phd_work/Models/BERT/BERT_8l_0.5/last'
    model_path = '/home/rfit/Telescope_Array/phd_work/Models/BERT/BERT_one_work/best'
    mask_model.load(model_path)
    return mask_model.eval()
def get_dataset(config, need_train_DS=False, many_val_loaders=True):
    """
    Предобработка: создание загрузчиков, модели, оптимизатора и логгера.

    Аргументы:
        config (dict): Конфигурация.
        need_train_DS (bool): Загрузить ли обучающую выборку.
        many_val_loaders (bool): Создавать ли несколько загрузчиков валидации.

    Возвращает:
        dict: Словарь с train_loader, val_loaders, model, optimizer, writer.
    """
    kwargs = DataSet.get_params_mask(config)
    kwargs['mc_params'] = True
    collate_fn = DataSet.wrapper_mask(DataSet.collate_fn_many_args, **kwargs)
    if need_train_DS:
        dataset = DataSet.VariableLengthDataset(config['data_path'], 'train',
                                                paticles=config['paticles']['train'],
                                                mc_params=True,
                                                reconstruction_params=config['reconstruction_params'],
                                                change_coordinat=config['change_coordinat'],
                                                change_sort=config['change_sort'],
                                                )
        train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    else:
        dataset = None
        train_loader = None
    # for many particles
    if many_val_loaders:
        val_loaders = []
        for p in config['paticles']['test']:
            val_dataset = DataSet.VariableLengthDataset(config['data_path'], 'test',
                                                        paticles=[p], mc_params=True,
                                                        reconstruction_params=config['reconstruction_params'],
                                                        change_coordinat=config['change_coordinat'],
                                                        change_sort=config['change_sort'],
                                                        )
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
            val_loaders.append(val_loader)
    else:
        val_dataset = DataSet.VariableLengthDataset(config['data_path'], 'test',
                                                    paticles=config['paticles']['test'],
                                                    mc_params=True,
                                                    reconstruction_params=config['reconstruction_params'],
                                                    change_coordinat=config['change_coordinat'],
                                                    change_sort=config['change_sort'],
                                                    )
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
        val_loaders = [val_loader]
    return val_loaders, train_loader
def random_mask(x: torch.Tensor, probability: float, mask_v: float = -11) -> torch.Tensor:
    # рызыгрывать вероятности а не индексы. 
    device = x.device

    token_mask = (x[:,:,0:1] != mask_v).to(device)  # [batch, maxlen]
    token_mask[:, 0] = False  # исключаем первый токен

    index = torch.sum(token_mask, dim=1)[:,0] # one dim -> batch
    token_mask[torch.arange(index.size(0)), index] = False
    
    probability_tensor = torch.rand_like(x[:,:,0:1]) # batch, len, 1
    probability_tensor = probability_tensor*token_mask # zero in supportive tokens
    token_mask = torch.where(probability_tensor>(1-probability), 1, 0).to(device)
    # to shape -> batch, len, 6
    token_mask = torch.repeat_interleave(token_mask, 6, dim=2).to(device).to(torch.bool)
    return token_mask
def make_TBoard(probability_list:List[float],
                prefix:Optional[str] = None):
    global_path = "/home/rfit/Telescope_Array/phd_work/TBruns/ClassificationPredict/"
    if prefix:
        global_path = os.path.join(global_path,prefix)
    TB_list = [SummaryWriter(log_dir=os.path.join(global_path, 'real')),
            #    SummaryWriter(log_dir=os.path.join(global_path, 'random'))
               ]
    
    for pr in probability_list:
        TB_list.append(SummaryWriter(log_dir=os.path.join(global_path, 'fake_pr_' + str(pr))))
        TB_list.append(SummaryWriter(log_dir=os.path.join(global_path, 'random_pr_' + str(pr))))
    ClassificationMetrics_list = []
    for tb in TB_list:
        metric = ClassificationMetrics(tb, 2)
        ClassificationMetrics_list.append(metric)
    return ClassificationMetrics_list
@torch.no_grad()
def run(mask_model, classificator, dataloaders,
            config, 
            particle:List[str] = ['pr', 'ph'],
            probability_list:List[float] = [0.1,0.5,0.7,1.0],):
    ClassificationMetrics_list = make_TBoard(probability_list)
    y_real_preds = None
    y_random_all = {}
    y_fake_preds_all = {}
    for pr in probability_list:
        y_fake_preds_all[pr] = None
        y_random_all[pr] = None
    y_target = None

    with torch.no_grad():
        for p, dataloader in enumerate(dataloaders):
            pbar_val = tqdm(dataloader, desc =f"VAL {particle[p]}")
            k=0
            for x, part, params_CR in pbar_val:
                k+=len(x)
                x = x.to(device)
                rand = torch.rand_like(x)
                part = torch.where(part == 1, 0, 1).to(device) # 0- photon, 1- proton  

                real_predict = classificator(x)
                # Initialize target only once per batch
                if y_real_preds is None:
                    y_real_preds = real_predict.detach().cpu().numpy()
                    y_target = part.detach().cpu().numpy()
                else:
                    y_real_preds = np.concatenate([y_real_preds, real_predict.detach().cpu().numpy()])
                    y_target = np.concatenate([y_target, part.detach().cpu().numpy()])
                
                # probability masking
                for probability in probability_list:
                    x_mask = random_mask(x, probability=probability, mask_v = config['padding_value']).to(device)
                    recon_x = mask_model(x, x_mask)
                    # Востанавливает только что замаскированно, значит надо вернуть реальные данные
                    # print(x_mask) # где True - там реконструкция. Уже стоит False на координатах
                    recon = torch.where(x_mask, recon_x, x)
                    recon_predict = classificator(recon)
                    random_x = torch.where(x_mask, rand, x).to(device)
                    random_predict = classificator(random_x)
                    
                    if y_fake_preds_all[probability] is None:
                        y_fake_preds_all[probability] = recon_predict.detach().cpu().numpy()
                        y_random_all[probability] = random_predict.detach().cpu().numpy()
                    else:
                        y_fake_preds_all[probability] = np.concatenate([y_fake_preds_all[probability], recon_predict.detach().cpu().numpy()])
                        y_random_all[probability] = np.concatenate([y_random_all[probability], random_predict.detach().cpu().numpy()])
        # metric calculate after processing all dataloaders
        real_metric = ClassificationMetrics_list[0]
        res = real_metric(y_real_preds, y_target)
        print('real', res)
        for i, probability in enumerate(probability_list):
            pr_str = str(probability)
            y_fake_preds = y_fake_preds_all[probability]
            y_random = y_random_all[probability]
            metric = ClassificationMetrics_list[i*2+1]
            res = metric(y_fake_preds, y_target)
            print('fake ', pr_str, res)
            metric = ClassificationMetrics_list[i*2+2]
            res = metric(y_random, y_target)
            print('random ', pr_str, res)
if __name__ == "__main__":
    classificator = init_class_model()
    mask_model = init_BERT()
    config = read_config('/home/rfit/Telescope_Array/phd_work/src/BERT/config.yaml')
    val_loaders, _ = get_dataset(config, many_val_loaders = False)
    run(mask_model, classificator, val_loaders, config,
            probability_list = [0.1,0.5,0.7,0.8,0.9,1.0])