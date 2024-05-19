import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns
import h5py
import tqdm
import zipfile
from tqdm.notebook import tqdm_notebook
import random
import pandas as pd
import time
import os
import math
from joblib import Parallel, delayed
import importlib
param_names=['signal','pl_fr','real_wf-pl_fr','mask']
def renorming(data,norm_params):
    val=np.array(list(norm_params.values()))
    max_c=val[:,0]
    min_c=val[:,1]
    data = data*(max_c-min_c) + min_c
#     if (i==0 and log):
#         data = data[:,:,:,i].assign(tf.math.exp(data[:,:,:,i])-1)
#     #         elif (i==1 or i==2):
#     #             data[:,:,:,i]=data[:,:,:,i]#*1e6
    return data
def norming(data,param_names=param_names,log=True):
    
    # ADD LOGARIFM FOR SIGNAL
    norm_params={}
    for i in range(data.shape[-1]):
        if (i==0 and log):
            data[:,:,:,i]=tf.math.log(data[:,:,:,i]+1)
        max_c=data[:,:,:,i].max()
        min_c=data[:,:,:,i].min()
        mean_c=data[:,:,:,i].mean()
        std_c=data[:,:,:,i].std()
        print('max_c,min_c',max_c,min_c)
        data[:,:,:,i]=(data[:,:,:,i]-min_c)/(max_c-min_c)

        norm_params[param_names[i]]=np.array([max_c,min_c])
    return norm_params
def im_one(i,data,axs):
    signal=data[i,:,:,0]
    real_time=data[i,:,:,1]+data[i,:,:,2]
    mask=data[i,:,:,3]
    sns.heatmap(data[i,:,:,0]*mask,annot=data[i,:,:,0]*mask,ax=axs[i,0],vmin=0.0, vmax=1.0)
    sns.heatmap(real_time*mask,annot=real_time*mask,ax=axs[i,1],vmin=-1.0, vmax=1.0)

def image_signal(data,norm_params,fake=None,dir_name=None,ep='not_ep',):
    n=data.shape[0]
    if (fake is None):
        fig, axs = plt.subplots(nrows=n, ncols=2, figsize=(10,n*5))
#         Parallel(n_jobs=-1)(delayed(im_one)(i,data,axs) for i in range(n))
        for i in range(n):
            im_one(i,data,axs)
            axs[i,0].set_title(f'number {str(i)}')
                           
        fig.suptitle('only real')
    else:
        fig, axs = plt.subplots(nrows=n, ncols=4, figsize=(10*2,n*5))
        data_after = renorming(data,norm_params)
        data_after = tf.where(tf.cast(tf.expand_dims(data_after[:,:,:,-1],axis=-1),tf.bool),data_after[:,:,:,:],np.nan)
        fake_after = renorming(fake,norm_params)
        fake_mask = tf.cast(tf.where(tf.cast(fake_after[:,:,:,3:4]>0.5,tf.bool),1,0),tf.float32)
        fake_after *= fake_mask
        fake_after = tf.where(tf.cast(tf.expand_dims(fake_after[:,:,:,-1],axis=-1),tf.bool),fake_after[:,:,:,:],np.nan)
        for i in range(n):
            sns.heatmap(data_after[i,:,:,0],annot=data_after[i,:,:,0],ax=axs[i,2],fmt=".1f")
            sns.heatmap(data_after[i,:,:,1],annot=data_after[i,:,:,2],ax=axs[i,3],fmt=".2f")
            
            
            sns.heatmap(fake_after[i,:,:,0],annot=fake_after[i,:,:,0],ax=axs[i,0],fmt=".1f")
            sns.heatmap(fake_after[i,:,:,1],annot=fake_after[i,:,:,2],ax=axs[i,1],fmt=".2f")
        fig.suptitle('fake     /    real')
    if dir_name:
        plt.savefig("{}/save_images/epoch{}.png".format(dir_name,ep))
        plt.close()
        
def images(cond_generator,num, data_all, generator,norm_params,
           noise_dim, dir_name='', ep='not_ep'):
    shape = (6,6,4)
    rand=np.random.choice(np.arange(len(data_all)),num)
    data_for_plot=np.zeros((num,shape[0],shape[1],shape[2]))
#     det_l = np.zeros((num,6,6,3))
#     theta_l = np.zeros((num,1))
#     phi_l = np.zeros((num,1))
#     courve_l = np.zeros((num,1))
#     S800_l = np.zeros((num,1))
    for i in range(num):
#         data, det, theta, phi, courve, S800 = data_all[rand[i]]
        data = data_all[rand[i]]
#         print(data.shape, det, theta, phi, courve, S800)
        data_for_plot[i]=data
#         det_l[i] = det
#         theta_l[i] = theta
#         phi_l[i] = phi
#         courve_l[i] = courve
#         S800_l[i] = S800
    noise = tf.random.normal(shape=(num,noise_dim))
#     fake=cond_generator([noise,det_l, theta_l, phi_l, courve_l,S800_l])
    fake=generator(noise)
    fake = np.array(fake)
    image_signal(data_for_plot,norm_params, fake=fake,dir_name=dir_name,ep=ep)