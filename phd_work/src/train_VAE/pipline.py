from torch.autograd import variable
from tqdm import tqdm
import argparse
import h5py as h5
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
import model as Model
import datasets as DataSet
import loss as Loss
from typing import Optional, Tuple, Union

from torch.utils.tensorboard import SummaryWriter

from metric import LatentMetric, DivedeMetrics
from utils import get_time, get_params_str, show_pred, read_config, clean_mask, MCPAR_index2srt, draw_logvar_mu
import logging
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

logger = logging.getLogger()
class Pipline():
    """
    Обучающий и инференс-пайплайн для вариационного автокодировщика (VAE),
    применяемого к данным с переменной длиной из эксперимента Telescope Array.

    Атрибуты:
        config (dict): Конфигурация, загруженная из YAML.
        model (nn.Module): Модель автокодировщика.
        optimizer (torch.optim.Optimizer): Оптимизатор.
        train_loader (DataLoader): DataLoader для обучающей выборки.
        val_loaders (List[DataLoader]): Список DataLoader-ов для валидации.
        writer (SummaryWriter): TensorBoard writer для логов.
        mask (int): Значение паддинга.
        koef_loss (torch.Tensor): Веса для функции потерь.
        scheduler (ReduceLROnPlateau): Планировщик скорости обучения.
    """

    def __init__(self, config, need_train_DS: bool = True, many_val_loaders: bool = True):
        """
        Инициализация пайплайна: загрузка конфигурации, данных, модели и оптимизатора.

        Аргументы:
            config (str): Путь к YAML-файлу конфигурации.
            need_train_DS (bool): Загружать ли обучающую выборку.
            many_val_loaders (bool): Создавать ли отдельный загрузчик для каждого типа частиц.
        """
        config = read_config(config)
        self.config = config
        self.prepipline_dict = self.prepipline(config, need_train_DS=need_train_DS, many_val_loaders= many_val_loaders)
        self.pretrain()
    def prepipline(self, config: dict, need_train_DS: bool = True, many_val_loaders: bool = True) -> dict:
        """
        Предобработка: создание загрузчиков, модели, оптимизатора и логгера.

        Аргументы:
            config (dict): Конфигурация.
            need_train_DS (bool): Загрузить ли обучающую выборку.
            many_val_loaders (bool): Создавать ли несколько загрузчиков валидации.

        Возвращает:
            dict: Словарь с train_loader, val_loaders, model, optimizer, writer.
        """
        name = config['PATH'].split('/')[-1]
        writer = SummaryWriter(log_dir=os.path.join('/home/rfit/Telescope_Array/phd_work/TBruns/', config['exp'], name))
        writer.add_text('hparams',  str(config))
        kwargs = DataSet.get_params_mask(config)
        kwargs['mc_params'] = True
        collate_fn = DataSet.wrapper_mask(DataSet.collate_fn_many_args, **kwargs)
        if need_train_DS:
            dataset = DataSet.VariableLengthDataset(config['data_path'], 'train',
                                                    paticles=config['paticles']['train'],
                                                    mc_params=True,
                                                    reconstruction_params=config['reconstruction_params'],
                                                    change_coordinat = config["change_coordinat"],
                                                    change_sort = config['change_sort'],
                                                    probability = config['paticles']['train_probability'],
                                                    recos=True)
            train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
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
                                                            change_coordinat=config.get("change_coordinat", False),
                                                            change_sort=config.get('change_sort', False),
                                                            recos=True)
                val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
                val_loaders.append(val_loader)
        else:
            val_dataset = DataSet.VariableLengthDataset(config['data_path'], 'test',
                                                        paticles=config['paticles']['test'],
                                                        mc_params=True,
                                                        reconstruction_params=config['reconstruction_params'],
                                                        change_coordinat=config.get("change_coordinat", False),
                                                        change_sort=config.get('change_sort', False),
                                                        recos=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
            val_loaders = [val_loader]        
        start_token = kwargs['start_token'].to(device)
        model = Model.VAE(config['input_dim'], config['hidden_dim'], config['latent_dim'], lstm2=config['lstm2'], start_token=start_token,
                          padding_value = config['padding_value'],
                          stop_token = config['stop_token'],
                          num_layers = config['num_layers'],
                          reconstruction_params = config['reconstruction_params'],
                          reparameterize_koef = config['reparameterize_koef'],
                          denoise_koef = config['denoise_koef'],
                          ).to(device)
        if config['chpt'] != 'None':
            model.load(config['chpt'])

        # differnet optimize

        # optimizer = optim.Adam(model.parameters(), lr=float(config['lr']))
        optimizer = optim.AdamW(model.encoder.parameters(), lr=float(config['lr']))
        optimizer_decoder = optim.AdamW(model.decoder.parameters(), lr=float(config['lr']))
        # write augmentes
        self.optimizer = optimizer
        self.optimizer_decoder = optimizer_decoder
        self.train_loader = train_loader
        self.model = model
        self.writer = writer
        print('split_num',config['split_num'], type(config['split_num']) )
        self.metric_lattent = LatentMetric(num_split=int(config['split_num']))
        self.metric_divide = DivedeMetrics(num_split=int(config['split_num']))
        self.metric_f1_best = 0
        return {'train_loader': train_loader, 'val_loaders': val_loaders, 'model': model, 'optimizer': optimizer, 'writer': writer}
    def load_chpt(self, chpt_path: str):
        """
        Загрузка весов модели из чекпоинта.

        Аргументы:
            chpt_path (str): Путь к файлу весов модели.
        """
        self.model.load(chpt_path)
    def pretrain(self):
        """
        Подготовка модели к обучению: пути, эпохи, веса, планировщик.
        """
        config = self.config
        PATH = os.path.join(config['save_model_path'], config['PATH'])
        self.PATH = PATH
        config['PATH'] = PATH
        print("Saving Path: {}".format(PATH))
        prepipline_dict = self.prepipline_dict
        self.train_loader = prepipline_dict['train_loader']
        self.val_loaders = prepipline_dict['val_loaders']
        self.model = prepipline_dict['model']
        self.optimizer = prepipline_dict['optimizer']
        self.writer = prepipline_dict['writer']
        self.epochs = config['epoches']
        self.mask = config['padding_value']
        self.show_index = config['show_index']
        self.koef_KL = np.zeros(self.epochs)
        # koef_KL:
        #     start: 0.0001
        #     end: 0.005
        #     start_it: 0
        #     end_it: 20
        self.koef_KL[config['koef_KL']['start_it']:config['koef_KL']['end_it']] = np.linspace(config['koef_KL']['start'], config['koef_KL']['end'], config['koef_KL']['end_it']-config['koef_KL']['start_it'])
        self.koef_KL[config['koef_KL']['end_it']:] = config['koef_KL']['end']
        self.koef_KL = torch.tensor(self.koef_KL).to(device)
        print('self.koef_KL', self.koef_KL)
        self.koef_DL = config['koef_DL']
        self.koef_mass = config['koef_mass']

        self.use_mask = config['use_mask']
        self.stop_token = config['stop_token']
        self.start_token = config['start_token']
        koef_loss = torch.tensor(config['koef_loss']).unsqueeze(0).to('cpu')
        self.paticles = self.config['paticles']
        self.koef_loss = koef_loss.to(device)
        self.koef_MMD = config['koef_MMD']
        os.makedirs(PATH, exist_ok = True)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.2, patience=5, threshold=0.005,)
    def train(self):
        """
        Обучает модель VAE и логирует метрики в TensorBoard.
        """
        self.loss_best = 1000
        iters = 0
        self.validation(epoch=0, analys=True)
        for epoch in range(self.epochs):
            print('lr_scheduler', self.optimizer.param_groups[0]['lr'])
            self.writer.add_scalar("lr_scheduler", self.optimizer.param_groups[0]['lr'], epoch)
            self.model.train()
            pbar = tqdm(self.train_loader, desc =f"TRAIN Epoch {epoch + 1}/{self.epochs}, Loss: 0.0")
            koef_KL = self.koef_KL[epoch]
            for x, part, params_CR, recos in pbar:  # x должен быть пакетом последовательностей с заполнением
                # x- data
                # part - promt mc_params in h5(look dataset.py)
                # params_CR by reconstruction index

                x = x.to(device)
                part = torch.where(part == 1, 0, 1).to(device) # 0- photon, 1- proton
                params_CR = params_CR.to(device) 
                self.optimizer.zero_grad()
                self.optimizer_decoder.zero_grad()
                recon_x, mu, log_var, pred_num, recon_pred = self.model(x)
                recon_loss, kl_divergence, num_det_loss, recon_params_loss, mmd_loss = Loss.vae_loss(recon_x, x, mu, log_var, pred_num,
                                                                                    recon_pred,
                                                                                    params_CR, 
                                                                                    part,
                                                                                    mask=self.mask,
                                                                                    use_mask=self.use_mask,
                                                                                    koef_loss=self.koef_loss

                                                                                    )
                num_det_loss *= self.koef_DL
                kl_divergence *= koef_KL
                recon_params_loss *= self.koef_mass
                mmd_loss *= self.koef_MMD
                loss = recon_loss + kl_divergence + num_det_loss + torch.mean(recon_params_loss) + mmd_loss
                self.writer.add_scalar("train/Loss", loss, iters)
                self.writer.add_scalar("train/KL_loss", kl_divergence, iters)
                self.writer.add_scalar("train/recon_loss", recon_loss, iters)
                self.writer.add_scalar("train/num_det_loss", num_det_loss, iters)
                self.writer.add_scalar("train/mmd_loss", mmd_loss, iters)
                
                loss.backward()
                self.optimizer.step()
                self.optimizer_decoder.step()
                pbar.set_description(f"TRAIN Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")
                iters += 1
            self.writer.add_scalar("train/koef_KL", koef_KL, epoch)
            self.validation(epoch=epoch, analys=True, koef_KL=koef_KL)

    def validation(self, epoch: Optional[int] = None, analys:bool=False, koef_KL: Optional[torch.Tensor] = None) -> None:
        """
        Performs validation on the model using the validation dataset.

        Parameters:
        epoch (int): The current epoch number.
        analys (bool): A flag indicating whether to perform analysis during validation.

        Returns:
        None. The function updates the model's state_dict if the validation loss is the best so far.
        It also prints the validation loss and loss_best, updates the learning rate scheduler,
        and writes the validation metrics to TensorBoard.
        """
        epoch = epoch if epoch is not None else -1
        model = self.model
        val_loaders = self.val_loaders
        koef_KL = koef_KL if koef_KL is not None else self.koef_KL[epoch]
        koef_DL = self.koef_DL

        model.eval()
        loss_mean = np.array([])
        KL_loss_mean = np.array([])
        recon_loss_mean = np.array([])
        num_det_loss_mean = np.array([])
        embading_dict = {}
        recon_loss_dict = {}
        KL_loss_dict = {}
        preds_log_var_dict = {}
        num_det_list = {}
        for i, val_loader in enumerate(val_loaders):
            particle = self.config['paticles']['test'][i]
            loss, KL, recon, num_det, embading, preds_log_var, preds_num_det, mmd_loss_list = self.validation_step(epoch=epoch, 
                                                                    val_loader=val_loader, 
                                                                    model=model,
                                                                    koef_KL=koef_KL, 
                                                                    koef_DL=koef_DL,
                                                                    particle=particle,
                                                                    reduce_loss_per_event=True,
                                                                    return_log_var_and_num_det=True)
            loss_mean = np.concatenate((loss_mean, loss))
            KL_loss_mean = np.concatenate((KL_loss_mean, KL))
            recon_loss_mean = np.concatenate((recon_loss_mean, recon))
            num_det_loss_mean = np.concatenate((num_det_loss_mean, num_det))
            # все что выше можно потом убрать 
            recon_loss_dict[particle] = recon
            KL_loss_dict[particle] = KL
            embading_dict[particle] = embading
            preds_log_var_dict[particle] = preds_log_var
            loss_final = loss_mean.mean()
            print(preds_log_var.shape, embading.shape,recon.shape, KL.shape)
            self.writer.add_scalar(f"val/Loss/all/{particle}", loss.mean(), epoch)
            self.writer.add_scalar(f"val/KL_loss/all/{particle}", KL.mean(), epoch)
            self.writer.add_scalar(f"val/recon_loss/all/{particle}", recon.mean(), epoch)
            self.writer.add_scalar(f"val/num_det_loss/{particle}", num_det.mean(), epoch)
            self.writer.add_scalar(f"val/mmd_loss_list/{particle}", mmd_loss_list.mean(), epoch)
            #draw logvar and mu
        fig = draw_logvar_mu(list(preds_log_var_dict.values()), 
                            list(embading_dict.values()), 
                            list(preds_log_var_dict.keys()))
        self.writer.add_figure(f"val/logvar_mu", fig, epoch)

        
        # write in TB
        # self.writer.add_scalar("val/Loss/all", loss_final, epoch)
        # self.writer.add_scalar("val/KL_loss/all", KL_loss_mean.mean(), epoch)
        # self.writer.add_scalar("val/recon_loss/all", recon_loss_mean.mean(), epoch)
        # self.writer.add_scalar("val/num_det_loss/all", num_det_loss_mean.mean(), epoch)
        metric_f1, fig= self.metric_lattent(embading_dict['pr'], embading_dict['photon'])
        _, fig_divide = self.metric_divide( recon_loss_dict['pr'], recon_loss_dict['photon'], 
                                            KL_loss_dict['pr'], KL_loss_dict['photon'],
                                            embading_dict['pr'], embading_dict['photon'],
                                            )
        self.writer.add_figure("val/lattent", fig, epoch)
        self.writer.add_figure("val/divide", fig_divide, epoch)
        self.writer.add_scalar("val/divide_f1", metric_f1, epoch)
        
        if analys:
            if epoch>0:
                self.scheduler.step(loss_final)
            if metric_f1>self.metric_f1_best:
                self.metric_f1_best = metric_f1
                torch.save(model.state_dict(), os.path.join(self.PATH, f'best'))
            torch.save(model.state_dict(), os.path.join(self.PATH, f'last'))

            print(f'Epoch {epoch + 1}, Loss: {loss_final} metric_f1 {self.metric_f1_best}')

    def validation_step(self, epoch: int, val_loader, model,
                    koef_KL=1, koef_DL=1, particle=None,
                    reduce_loss_per_event:bool = False, 
                    return_log_var_and_num_det:bool = False,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Одна итерация валидации по одному типу частиц.

        Аргументы:
            epoch (int): Номер текущей эпохи.
            val_loader (DataLoader): DataLoader с тестовыми данными.
            model (nn.Module): Модель.
            koef_KL (float): Вес для KL-дивергенции.
            koef_DL (float): Вес для потерь по числу детекторов.
            particle (str, optional): Имя типа частицы.

        Возвращает:
            Кортеж numpy-массивов: полные потери, KL, реконструкция и число детекторов.
        """
        particle = '' if particle is None else particle
        loss_mean = []
        KL_loss_mean = []
        recon_loss_mean = []
        num_det_loss_mean = []
        recon_params_loss_mean = [0]*len(self.config['reconstruction_params'])
        embading = np.zeros((0, self.config['latent_dim']))
        pbar_val = tqdm(val_loader, desc =f"VAL Epoch {epoch + 1} in {particle}, Loss: 0.0")
        num_examples = 0
        preds_log_var = np.zeros((0,self.config['latent_dim']))
        preds_num_det = np.zeros((0, 1))
        mmd_loss_list = []
        for x, part, params_CR, recos in pbar_val:  # x should be a batch of sequences with padding
            num_examples += x.size(0)

            with torch.no_grad():
                x = x.to(device)
                part = torch.where(part == 1, 0, 1).to(device) # 0- photon, 1- proton
                params_CR = params_CR.to(device) 
                recon_x, mu, log_var, pred_num, recon_pred = model(x)
                recon_loss, kl_divergence, num_det_loss, recon_params_loss, mmd_loss = Loss.vae_loss(recon_x, x, mu, log_var, pred_num,
                                                                                        recon_pred,
                                                                                        params_CR, 
                                                                                        part,
                                                                                        mask=self.mask,
                                                                                        use_mask=self.use_mask,
                                                                                        koef_loss=self.koef_loss,
                                                                                        reduce_loss_per_event=reduce_loss_per_event

                                                                                        )
                embading = np.concatenate((embading, mu.cpu().detach().numpy()), axis=0)
                preds_log_var = np.concatenate((preds_log_var, log_var.cpu().detach().numpy()), axis=0)
                preds_num_det = np.concatenate((preds_num_det, pred_num.cpu().detach().numpy()), axis=0)
                
                # kl_divergence *= koef_KL
                # num_det_loss *= koef_DL
                # recon_params_loss *= self.koef_mass
                if reduce_loss_per_event:
                    recon_loss_mean.append(recon_loss.to('cpu').numpy())
                    KL_loss_mean.append(kl_divergence.to('cpu').numpy())
                    num_det_loss_mean.append(num_det_loss.to('cpu').numpy())
                    mmd_loss_list.append(mmd_loss.cpu().detach().numpy())
                
                    # Вычисляем общий loss для каждого события
                    if len(recon_params_loss.shape) == 1:
                        # recon_params_loss уже (batch,)
                        total_loss = recon_loss + kl_divergence*koef_KL + num_det_loss*koef_DL + recon_params_loss*self.koef_mass +mmd_loss*self.koef_MMD
                    else:
                        # recon_params_loss (batch, params), усредняем по params
                        recon_params_loss_mean_per_event = torch.mean(recon_params_loss, dim=1)  # (batch,)
                        total_loss = recon_loss + kl_divergence*koef_KL + num_det_loss*koef_DL + recon_params_loss_mean_per_event*self.koef_mass + mmd_loss*self.koef_MMD
                    loss_mean.append(total_loss.to('cpu').numpy())
                else:
                    loss = recon_loss + kl_divergence*koef_KL + num_det_loss*koef_DL + torch.mean(recon_params_loss)*self.koef_mass
                    loss_mean.append(loss.item())
                    KL_loss_mean.append(kl_divergence.item())
                    recon_loss_mean.append(recon_loss.to('cpu').item())
                    num_det_loss_mean.append(num_det_loss.item())
                    mmd_loss_list.append(mmd_loss.cpu().detach().numpy().item())
                    for i,par in enumerate(recon_params_loss_mean): 
                        recon_params_loss_mean[i] += recon_params_loss[i].to('cpu').item()
                pbar_val.set_description(f"VAL Epoch {epoch + 1} in {particle}")
        #show from last batch
        real = x[self.show_index]
        print(recon_x.shape)
        fake = recon_x[self.show_index]
        num = pred_num[self.show_index]
        for ii in range(len(self.show_index)):
            # get from back side
            i = -ii
            fig = show_pred(real[i], fake[i], tokens = (self.start_token, self.stop_token, self.mask), lenght_predict = num[i])
            self.writer.add_figure(f"val/show_pred_{ii}/{particle}", fig, epoch)
        
        if reduce_loss_per_event:
            # Конкатенируем все батчи в один массив для каждого события
            loss_mean = np.concatenate(loss_mean) if len(loss_mean) > 0 else np.array([])
            recon_loss_mean = np.concatenate(recon_loss_mean) if len(recon_loss_mean) > 0 else np.array([])
            KL_loss_mean = np.concatenate(KL_loss_mean) if len(KL_loss_mean) > 0 else np.array([])
            num_det_loss_mean = np.concatenate(num_det_loss_mean) if len(num_det_loss_mean) > 0 else np.array([])
            mmd_loss_list = np.array(mmd_loss_list) if len(mmd_loss_list) > 0 else np.array([])
            if return_log_var_and_num_det:
                return loss_mean, KL_loss_mean, recon_loss_mean, num_det_loss_mean, embading, preds_log_var, preds_num_det, mmd_loss_list
            else:
                return loss_mean, KL_loss_mean, recon_loss_mean, num_det_loss_mean, embading, mmd_loss_list
        # recon_params_loss_mean = torch.concat(recon_params_loss_mean, dim=1).numpy()
        # recon_params_loss_mean = np.array(recon_params_loss_mean)/num_examples
        # loss_final = np.array(loss_mean).mean()
        # # write in TB
        # self.writer.add_scalar(f"val/Loss/{particle}", loss_final, epoch)
        # self.writer.add_scalar(f"val/KL_loss/{particle}", np.array(KL_loss_mean).mean(), epoch)
        # self.writer.add_scalar(f"val/recon_loss/{particle}", np.array(recon_loss_mean).mean(), epoch)
        # self.writer.add_scalar(f"val/num_det_loss/{particle}", np.array(num_det_loss_mean).mean(), epoch)
        # if self.config['reconstruction_params'] is not None:
        #     for i, ind in enumerate(self.config['reconstruction_params']):
        #         name_param = MCPAR_index2srt(ind)
        #         self.writer.add_scalar(f"val/{name_param}/{particle}", recon_params_loss_mean[i].mean(), epoch)

        # return np.array(loss_mean), np.array(KL_loss_mean), np.array(recon_loss_mean), np.array(num_det_loss_mean), embading
    def get_num_det_list(self, val_loader):
        num_det_list = np.array([])
        for x, part, *_ in val_loader:
            x = x.to(device)
            num_det = Loss.calc_det(x,self.mask,use_mask=False).to('cpu').detach().numpy()
            # return array like [7 7 7 7 7 7] - теперь num_det уже имеет форму (batch,)
            num_det_list = np.concatenate((num_det_list, num_det))
        return num_det_list

    def test(self, model_path: str = None):
        """
        Тестируем модель. Смотрим на разделение частиц по функции ошибок

        1 - строим графики в зависимости от кол-ва детекторов в событии
        2 - записываем латеные представления, ошики и кол-во детекторов в событии в csv файл
        """
        path_save = '/home/rfit/Telescope_Array/phd_work/src/train_VAE/info_Transfoemr_Kharuk_one_work_photon_0.01_FullyConnected_4'
        model = self.model
        if model_path is not None:
            model.load(model_path)
        val_loaders = self.val_loaders
        koef_KL = self.koef_KL
        koef_DL = self.koef_DL


        for i, val_loader in enumerate(val_loaders):
            mode = ''
            model.eval()
            KL_loss_mean = np.zeros(0)
            num_det_list = np.zeros(0)
            recon_loss_mean = np.zeros(0)
            num_det_loss_mean = np.zeros(0)
            recon_params_loss_mean = [0]*len(self.config['reconstruction_params'])
            embading = np.zeros((0, self.config['latent_dim']))
            preds_log_var = np.zeros((0,self.config['latent_dim']))
            preds_num_det = np.zeros((0, 1))
            mmd_loss_list = np.zeros(0)
            recos_list = []
            particle = self.config['paticles']['test'][i]
            path = os.path.join(path_save, particle)
            os.makedirs(path, exist_ok=True)
            with torch.no_grad():
                for x, part, params_CR, recos in tqdm(val_loader):  # x should be a batch of sequences with padding
                
                    x = x.to(device)
                    part = torch.where(part == 1, 0, 1).to(device) # 0- photon, 1- proton
                    params_CR = params_CR.to(device) 
                    num_det = Loss.calc_det(x,self.mask,use_mask=False).to('cpu').detach().numpy()
                    num_det_list = np.concatenate((num_det_list, num_det), axis=0)
                    recon_x, mu, log_var, pred_num, recon_pred = model(x)
                    recon_loss, kl_divergence, num_det_loss, recon_params_loss, mmd_loss = Loss.vae_loss(recon_x, x, mu, log_var, pred_num,
                                                                                            recon_pred,
                                                                                            params_CR, 
                                                                                            part,
                                                                                            mask=self.mask,
                                                                                            use_mask=self.use_mask,
                                                                                            koef_loss=self.koef_loss,
                                                                                            reduce_loss_per_event=True,

                                                                                            )
                    recos_list.append(recos)
                    embading = np.concatenate((embading, mu.cpu().detach().numpy()), axis=0)
                    preds_log_var = np.concatenate((preds_log_var, log_var.cpu().detach().numpy()), axis=0)
                    preds_num_det = np.concatenate((preds_num_det, pred_num.cpu().detach().numpy()), axis=0)
                    recon_loss_mean = np.concatenate((recon_loss_mean, recon_loss.to('cpu').numpy()), axis=0)
                    KL_loss_mean = np.concatenate((KL_loss_mean, kl_divergence.to('cpu').numpy()), axis=0)
                    num_det_loss_mean = np.concatenate((num_det_loss_mean, num_det_loss.to('cpu').numpy()), axis=0)
                    # mmd_loss_list = np.concatenate((mmd_loss_list, mmd_loss.cpu().detach().numpy()), axis=0)
                
                print(f'num_det_list.shape {num_det_list.shape}, recon_loss_mean.shape {recon_loss_mean.shape}, KL_loss_mean.shape {KL_loss_mean.shape}, num_det_loss_mean.shape {num_det_loss_mean.shape}, mmd_loss_list.shape {mmd_loss_list.shape}, embading.shape {embading.shape}, preds_log_var.shape {preds_log_var.shape}, preds_num_det.shape {preds_num_det.shape}')
                np.save(os.path.join(path, mode, f'num_det.npy'), num_det_list)
                np.save(os.path.join(path, mode, f'recon_loss.npy'), recon_loss_mean)
                np.save(os.path.join(path, mode, f'embading.npy'), embading)
                np.save(os.path.join(path, mode, f'loss_num_det.npy'), num_det_loss_mean)
                np.save(os.path.join(path, mode, f'preds_log_var.npy'), preds_log_var)
                np.save(os.path.join(path, mode, f'preds_num_det.npy'), preds_num_det)
                np.save(os.path.join(path, mode, f'recos.npy'), np.concatenate(recos_list, axis=0))
    def predict(self, test_loader, i, 
        NoneLoss: bool = False, 
        ):
        """
        Предсказывает латентные представления и реконструкции.
        Аргументы:
            NoneLoss (bool): Если True, не усреднять потери.

        Возвращает:
            Кортеж: латенты, реконструкции, метки частиц, потери реконструкции.
        """
        model = self.model
        model.eval()
        latent_list = []
        recon__list = []
        particles = []
        all_loss = None
        test_loaders = self.val_loaders
        dict_info = {}
        with torch.no_grad():
            # for i, test_loader in enumerate(test_loaders):
            for x, part, params_CR, *_ in tqdm(test_loader):
                x = x.to(device)
                part = torch.where(part == 1, 0, 1).to(device) # 0- photon, 1- proton
                params_CR = params_CR.to(device)
                mu, log_var, (h_n, c_n) = model.encoder(x)
                recon_x, mu, log_var, pred_num, pred_mass = model(x)
                if not(NoneLoss):
                    # Можно использовать если понадобятся другие лоссы
                    recon_loss, kl_divergence, num_det_loss, mass_loss, mmd_loss = Loss.vae_loss(recon_x, x, mu, log_var, pred_num, pred_mass, params_CR, part,
                                                                                    mask=self.mask, use_mask=self.use_mask, koef_loss=self.koef_loss,
                                                                                    reduce_loss_per_event = True
                                                                                    )
                    if all_loss is None:
                        all_loss = recon_loss.cpu().detach() 
                    else:
                        all_loss = torch.cat((all_loss, recon_loss.cpu().detach()))
                else:
                    recon_loss, kl_divergence, num_det_loss, mass_loss, mmd_loss = Loss.vae_loss_none(recon_x, x, mu, log_var, pred_num, recon_pred=pred_mass, params_CR=params_CR,
                                                                                    mask=self.mask, use_mask=self.use_mask, koef_loss=self.koef_loss,
                                                                                    )
                    if all_loss is None:
                        all_loss = []
                        all_loss.append(recon_loss.cpu().detach())
                    else:
                        all_loss.append(recon_loss.cpu().detach())
                latent_list.append(mu.cpu())
                recon__list.append(recon_x.cpu())
                # params.append(par.cpu())
                particles += [self.config['paticles']['test'][i]] * mu.shape[0] # for equal lenght with latent

                #write in dict
                # TODO otimize
                try:
                    dict_info[self.config['paticles']['test'][i]] = torch.cat((dict_info[self.config['paticles']['test'][i]], mu.cpu().detach() ), dim=0)
                except KeyError:
                    dict_info[self.config['paticles']['test'][i]] = mu.cpu().detach() 
        latent_list = torch.cat(latent_list, dim=0)
        # variable lenght. So this is not wor
        return latent_list, recon__list, particles, all_loss#params

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="A simple example of argparse")

    # Add optional arguments
    parser.add_argument("-m", "--mode", type=str, help="The output file to save results", default="train")
    parser.add_argument("-e", "--write_embading", type=bool, help="Write latent data in TB for project analys", default="True")
    # Parse the arguments
    args = parser.parse_args()

    config = 'config.yaml'
    if args.mode == 'train':
        pipline = Pipline(config)
        print('TRAIN PIPLINE')
        pipline.train()
    elif args.mode == 'test':
        pipline = Pipline(config, need_train_DS=False)
        print('TEST PIPLINE')
        pipline.test(model_path='/home/rfit/Telescope_Array/phd_work/Models/AutoEncoder/info_Transfoemr_Kharuk_one_work_photon_0.01_FullyConnected_4/last')
    elif args.mode == 'latent':
        pipline = Pipline(config, need_train_DS=False)
        print('Latent PIPLINE')
        pipline.predict_latent(args.write_embading)
    elif args.mode == 'grid_search':
        pipline = Pipline(config)
        print('TRAIN PIPLINE')

        # variable 
        variable = {'koef_KL': {'start': 1e-3, 'finish': 1e-1, 'step': 2, 'mode': 'mul'}}
        