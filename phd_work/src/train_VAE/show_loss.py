import h5py as h5
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, roc_auc_score
import loss as Loss
from typing import Tuple
import pipline
def plot_roc_curve( loss_pr, loss_ph, ax):
    """
    Строит ROC кривую и вычисляет AUC
    
    Args:
        y_true: истинные метки (0 или 1)
        y_scores: вероятности положительного класса (второй столбец из softmax)
        epoch: номер эпохи для логирования
    """
    label_pr = np.zeros_like(loss_pr)
    label_ph = np.ones_like(loss_ph)
    label = np.concatenate((label_pr, label_ph))
    loss = np.concatenate((loss_pr, loss_ph))
    # Вычисляем ROC кривую
    fpr, tpr, thresholds = roc_curve(label, loss)
    roc_auc = auc(fpr, tpr)
    
    # Создаем график
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random classifier (AUC = 0.5)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return ax
data_path = '/home3/rfit/Telescope_Array/phd_work/data/normed/pr_photon_0001_excl_sat_T_excl_geo_T_one_work.h5'
pip = pipline.Pipline(config = 'config.yaml')
pip.load_chpt(chpt_path = '/home/rfit/Telescope_Array/phd_work/Models/AutoEncoder/small_decoder_VAE/last')




self = pip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def validation_step(self, epoch: int, val_loader, model,
                koef_KL=1, koef_DL=1, particle=None,
                reduce_loss_per_event = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    recon_all = np.zeros((0))
    kl_all = np.zeros((0))
    pbar_val = tqdm(val_loader, desc =f"VAL ")
    num_examples = 0
    for x, part, params_CR in pbar_val:  # x should be a batch of sequences with padding
        num_examples += x.size(0)

        with torch.no_grad():
            x = x.to(device)
            part = torch.where(part == 1, 0, 1).to(device) # 0- photon, 1- proton
            params_CR = params_CR.to(device) 
            recon_x, mu, log_var, pred_num, recon_pred = model(x)
            recon_loss, kl_divergence, num_det_loss, recon_params_loss = Loss.vae_loss(recon_x, x, mu, log_var, pred_num,
                                                                                    recon_pred,
                                                                                    params_CR, 
                                                                                    part,
                                                                                    mask=self.mask,
                                                                                    use_mask=self.use_mask,
                                                                                    koef_loss=self.koef_loss,
                                                                                    reduce_loss_per_event = reduce_loss_per_event

                                                                                    )
            embading = np.concatenate((embading, mu.cpu().detach().numpy()), axis=0)
            # kl_divergence *= koef_KL
            # num_det_loss *= koef_DL
            # recon_params_loss *= self.koef_mass
            print(recon_loss.shape, kl_divergence)
            recon_all = np.concatenate((recon_all, recon_loss.to('cpu').numpy()), axis=0)
            # kl_all = np.concatenate((kl_all, kl_divergence.to('cpu').numpy()), axis=0)
            pbar_val.set_description(f"VAL ")
    # recon_params_loss_mean = torch.concat(recon_params_loss_mean, dim=1).numpy()

    return np.array(loss_mean), kl_all, recon_all, np.array(num_det_loss_mean), embading
    

model = self.model
val_loaders = self.val_loaders
koef_KL = self.koef_KL
koef_DL = self.koef_DL

model.eval()
loss_mean = np.array([])
KL_loss_mean = np.array([])
recon_loss_mean = np.array([])
num_det_loss_mean = np.array([])
embading_dict = {}
recon_loss_dict = {}
KL_loss_dict = {}
for i, val_loader in enumerate(val_loaders):
    particle = self.config['paticles']['test'][i]
    loss, KL, recon, num_det, embading = validation_step(self, epoch=-1, val_loader=val_loader, model=model, koef_KL=koef_KL, koef_DL=koef_DL,particle=particle,reduce_loss_per_event  =True )
    # loss_mean = np.concatenate((loss_mean, loss))
    # KL_loss_mean = np.concatenate((KL_loss_mean, KL))
    # recon_loss_mean = np.concatenate((recon_loss_mean, recon))
    # num_det_loss_mean = np.concatenate((num_det_loss_mean, num_det))
    # все что выше можно потом убрать 
    recon_loss_dict[particle] = recon
    KL_loss_dict[particle] = KL
    embading_dict[particle] = embading
fig, ax = plt.subplots(1,2,figsize = (10,5))
plot_roc_curve(recon_loss_dict['pr'], recon_loss_dict['photon'], ax[0])
ax[1].hist(recon_loss_dict['pr'], label='proton', log=True, histtype = 'step', density =True)
ax[1].hist(recon_loss_dict['photon'], label='photon',log=True, histtype = 'step', density =True)
ax[1].legend()
ax[1].set_title('Loss hist')
ax[1].set_xlabel('loss')
ax[1].set_ylabel('log(num)')
mean_pr = recon_loss_dict['pr'].mean()
mean_ph = recon_loss_dict['photon'].mean()
std_pr = recon_loss_dict['pr'].std()
std_ph = recon_loss_dict['photon'].std()
plt.text(0.7, 0.7, f'pr mean {mean_pr.mean():.3f} \nphoton mean {mean_ph.mean():.3f} \npr std {std_pr.mean():.3f} \nphoton std {std_ph.mean():.3f}', fontsize=10)
plt.savefig('loss_hist.png')