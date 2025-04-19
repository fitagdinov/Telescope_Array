
import h5py as h5
import numpy as np
import os
import os
os.chdir('/home/rfit/Telescope_Array/phd_work/src/train_VAE/')
print(os.listdir())
import sys
sys.path.append('/home/rfit/Telescope_Array/phd_work/src/train_VAE/')
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pipline
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from tqdm import tqdm
from sklearn.neighbors import KernelDensity
f_output = open("hists/output.txt", "a")
# data_path = '/home3/rfit/Telescope_Array/phd_work/data/normed/pr_fe_q4_e1_0110_excl_sat_F_excl_geo_F.h5'
data_path = '/home3/rfit/Telescope_Array/phd_work/data/normed/pr_photon_q4_e1_0110_excl_sat_F_excl_geo_F.h5'
data_path = '/home3/rfit/Telescope_Array/phd_work/data/normed/pr_photon_0001_excl_sat_F_excl_geo_F.h5'


with h5.File(data_path,'r') as f:
    print('keys', list(f.keys()))
    test = f['test']
    keys = list(test.keys())
    mc_params = test['mc_params'][:]
    for k in keys:
        print(k, test[k].shape)

config = 'config.yaml'
model = pipline.Pipline(config)
model.config['latent_dim'] = 8
exp = 'Fe_Pr18.01.2025_15:29latent_dim=16;_hidden_dim=512;_use_mask=True;_'
chpt = "../../Models/AutoEncoder/Proton_train_lat=8/best"
model.load_chpt(chpt)
latent_list, params, loss = model.predict_latent(write_embedding=False, choise_num = 100000)


select_dict = {}
set_parasm = set(params)
for i in tqdm(range(len(params))):
    try:
        select_dict[params[i]] = torch.concat([select_dict[params[i]], latent_list[i:i+1]], dim=0)
    except KeyError:
        select_dict[params[i]] = latent_list[i:i+1]


X = select_dict['pr'][:60000]
X_test = select_dict['pr'][60000:]
Y = select_dict['photon']

for bandwidth in tqdm(np.arange(0.4 ,5.5,0.2)):
    print(f'start bandwidth={bandwidth}', file = f_output)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X)
    log_dens = kde.score_samples(X)
    log_dens_Y = kde.score_samples(Y)
    log_dens_pr_test = kde.score_samples(X_test)

    th_01 = np.quantile(np.exp(log_dens), 0.1)
    th_90 = np.quantile(np.exp(log_dens), 0.9)
    th_01_pr = np.quantile(np.exp(log_dens_pr_test), 0.1)
    th_90_pr = np.quantile(np.exp(log_dens_pr_test), 0.9)
    less_01 = sum(np.exp(log_dens_Y)<th_01)
    more_90 = sum(np.exp(log_dens_Y)>th_90)
    less_01 = round(less_01/len(log_dens_Y),3)
    more_90 = round(more_90/len(log_dens_Y),3)
    
    print(th_01, th_01_pr, th_90, th_90_pr)
    print(less_01, more_90, file = f_output)
    #hists
    fig, axs = plt.subplots(4,1, figsize=(10,40))
    x_lim=0.002
    axs[0].hist(np.exp(log_dens), bins=100, color=(0,1,0,0.1),density=True, label = "Train")
    axs[1].hist(np.exp(log_dens_Y), bins=100, color=(0,0,0,0.5),density=True, label = "photon")
    axs[2].hist(np.exp(log_dens_pr_test), bins=100, color=(1,0,0,0.5), density=True, label = "Test_pr")
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[0].set_title(f'train_pr, th01 {th_01:.3g} th90 {th_90:.3g}')
    axs[1].set_title(f'test_photon, th01 {th_01_pr:.3g} th90 {th_90_pr:.3g}')

    axs[2].set_title('test_pr')
    
    
    axs[3].hist(np.exp(log_dens), bins=100, color=(0,1,0,0.1),density=True, label = "Train")
    axs[3].hist(np.exp(log_dens_Y), bins=100, color=(0,0,0,0.5),density=True, label = "photon")
    axs[3].hist(np.exp(log_dens_pr_test), bins=100, color=(1,0,0,0.5), density=True, label = "Test_pr")
    axs[3].legend()
    axs[3].set_title(f'all less {less_01} more {more_90}')
    plt.savefig(f'hists/bandwidth={bandwidth}.png')
    del fig
    print(f'end bandwidth={bandwidth}', file = f_output)
    plt.close()
f_output.close()