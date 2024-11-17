import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns
import h5py
import tqdm
import zipfile
from tqdm.notebook import tqdm_notebook
from progress.bar import IncrementalBar
import random
import pandas as pd
import time
import os
import math
from joblib import Parallel, delayed
import importlib

from utils import utils
from utils import train_utils
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    # tf.config.set_visible_devices(gpu, 'GPU')
path_new = '/home3/ivkhar/TA/data/MC/bundled/pr_q3_14yr_1745_0010_excl-sat_F_excl-geo_F_take-log-wf-False_bundled.h5'
path = path_new
num=-1
print(f"READ DATA from {path_new}")
with h5py.File(path,'r') as f:
    print(f.keys())
    data=f['dt_bunlde'][:num,:,:,3:7]
    print(data.shape)
    detectors_rub = f['dt_bunlde'][:num,:,:,:3] * 1.2 #/ 6 # norming
    real_ang = f['mc_params'][:num,4:6]
    recos = f['recos'][:num]
    dt_params =f['dt_params'][:num]
    ev_starts = f['ev_starts'][:num]
    if path == path_new:
        dt_bunlde_mask = f['dt_bunlde_mask'][:num]
norm_params=utils.norming(data,log=False)
theta = tf.cast(real_ang[:,0:1]/180*3.1415,tf.float32)
phi = tf.cast(real_ang[:,1:2]/180*3.1415,tf.float32)
courve  = tf.cast(recos[:,6:7],tf.float32)
S800 = tf.cast(recos[:,2:3],tf.float32)
chi_rub = tf.cast(recos[:,5:6],tf.float32)

print("END READ DATA")

boundaries = [1000]
values = [ 0.0001,  0.0001/5]
learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)

print("SELECT DATA WITH ANGLE")
ind = tf.where(real_ang[:,0]>20)[:,0]
data = tf.gather(data,ind)
detectors_rub = tf.gather(detectors_rub,ind)
dt_bunlde_mask  = tf.gather(dt_bunlde_mask,ind)
print(data.shape,ind.shape,detectors_rub.shape,dt_bunlde_mask.shape)

n=int(0.9*len(data))
train = data[:n]
test =data[n:]
detectors_rub = detectors_rub[:n]
dt_bunlde_mask =dt_bunlde_mask[:n]
print(train.shape,detectors_rub.shape,dt_bunlde_mask.shape)
print("END SELECT DATA WITH ANGLE")

print("LOAD MODEL")
noise_dim=50
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)

generator=tf.keras.models.load_model('../Models/deep_5/save_model/generator/ep'+str(100))
discriminator=tf.keras.models.load_model('../Models/deep_5/save_model/discriminator/ep'+str(100))

def func_chunks_generators(lst, n):
    '''передается масив и число.масив разбивается на масивы длиной не более n
    пример func_chunks_generators([1,2,3,4,5], 3) -> [[1,2,3],[4,5]]
    lst- масив
    n- число, пределяющее максимальную длину'''
    l=[]
    for i in range(0, len(lst), n):
         l.append(lst[i : i + n])
    return(l)
batch=100
epochs=250
ep_start=0
g_list=[]
d_list=[]
time_list=[]

def train_WGAN(epochs,train_data,test_data,batch,dir_name,
               detectors_rub,
              ):
    lamda = tf.constant(5,dtype=tf.float32)
    weight_gp = tf.constant(10,dtype=tf.float32)
    if not(os.path.exists(dir_name)):
        os.mkdir(dir_name)
        os.mkdir(os.path.join(dir_name,'save_images'))
        os.mkdir(os.path.join(dir_name,'save_model'))
        os.mkdir(os.path.join(dir_name,'save_model/discriminator'))
        os.mkdir(os.path.join(dir_name,'save_model/generator'))
    train_data = func_chunks_generators(data, batch)
    detectors_rub_step = func_chunks_generators(detectors_rub, batch)
    dt_bunlde_mask_step = func_chunks_generators(dt_bunlde_mask, batch)
    for j in tqdm_notebook(range(ep_start,epochs),'ep'):
        for num in tqdm_notebook(range(0,len(train_data)-300),f'epoch num{j}'):
            step_data=tf.Variable(train_data[num],dtype = tf.float32)
            g,d,t,r=train_utils.train_step_WGAN(5, batch, step_data, weight_gp,detectors_rub_step[num],dt_bunlde_mask_step[num],
                                                noise_dim, generator, discriminator, discriminator_optimizer, generator_optimizer, 
                                                learning_rate_fn, norm_params
                                                )
            d_list.append(d)
            g_list.append(g)
#             time_list.append(t)
        if (j % 10 == 0):
            utils.images(generator,num=10, data_all=test_data,
                   noise_dim=noise_dim,dir_name=dir_name,ep=j)
            discriminator.save("{}/save_model/discriminator/ep{}".format(dir_name,j))
            generator.save("{}/save_model/generator/ep{}".format(dir_name,j))
            plt.close()
dir_name = '../Models/Conditional+Phys_loss'
print("START TRAIN")
train_WGAN(epochs,train,test,batch,dir_name=dir_name,detectors_rub=detectors_rub)