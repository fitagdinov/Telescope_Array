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
import reconstruction as reco
import utils as Utils
def discriminator_loss(real_output, fake_output):
    real_loss = -tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
def reco_loss(data,detectors_rub,dt_bunlde_mask,learning_rate_fn,norm_params):
    data_ = Utils.utils.renorming(data,norm_params)
    chi_list, params_list = reco.optimization_2(data_,iterats=500,num=None,
                                          detectors_rub=detectors_rub,
                                          add_mask = dt_bunlde_mask,
                                          use_L = False,
                                          use_core =False,
#                                           S800_rub=S800,
                                          optim_name="SGD",l_r =learning_rate_fn,
#                                           use_L3=False,
                                            find_core=False,
                                         )
    return chi_list[-1,:,0]
def loss_time(data,kof=10,res=tf.Variable(0,dtype=tf.float32)):
#     res=tf.Variable(0,dtype=tf.float32)
    for i in range(data.shape[0]):
        data_n=tf.where(tf.cast(tf.expand_dims(data[i,:,:,3],axis=-1),tf.bool),data[i],np.nan)
        arg=tf.where(data_n[:,:,0] == tf.math.reduce_max(data[i,:,:,0]))[0]
        time=data_n[arg[0],arg[1],2]
        min_time=tf.math.reduce_min(tf.where(tf.math.is_nan(data_n[:,:,2]),np.inf,data_n[:,:,2]))
        res=res+time-min_time
    return(res/data.shape[0]*kof)
        
#     max_active_time=
def generator_loss(fake_output):
#     return loss_function(tf.ones_like(fake_output), fake_output)
    return -tf.reduce_mean(fake_output)

def gradient_penalti(batch,real_data,fake_data,discriminator):
#   alpha = tf.random.normal([batch, 1], 0.0, 1.0)
#   diff = fake_data - real_data
#   interpolated = real_data + alpha * diff
    epsilon=tf.random.uniform(shape=(batch,1,1,1),dtype=tf.dtypes.float32)
#     print(epsilon)
    interpolated=real_data-epsilon*(real_data-fake_data)# вычисление x^ как в статье
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred=discriminator(interpolated,training=True)  # D(x^)
    grads = gp_tape.gradient(pred, [interpolated])[0]# because list
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp    
# @tf.function
def train_step_WGAN(labda,batch,real_data,weight_gp,detectors_rub,dt_bunlde_mask,
                    noise_dim, generator,discriminator,discriminator_optimizer,generator_optimizer,
                    learning_rate_fn,norm_params):
  # labda --> number learling critic
  #weight --> weight gradient_penalti

  #learning critic
    real_time = real_data[:,:,:,1:2] +real_data[:,:,:,2:3]
    mask = real_data[:,:,:,3:4]
    t0=tf.constant(0, dtype=tf.float32)
    for i in range(labda):
        with tf.GradientTape() as gr:
            noise = tf.random.normal(shape=(batch,noise_dim))
            fake_data=generator(noise)
            real_data=real_data
            fake_predict=discriminator(fake_data)
            real_predict=discriminator(real_data)
            real_data=tf.cast(real_data,dtype=tf.float32)
            gp=gradient_penalti(batch,real_data,fake_data,discriminator)
            disc_loss=discriminator_loss(real_predict,fake_predict)+weight_gp*gp
        d_grad=gr.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(d_grad, discriminator.trainable_variables))
  #learning generator
    noise = tf.random.normal(shape=(batch,noise_dim))
    with tf.GradientTape() as gr:
        fake_data=generator(noise)
        fake_predict=discriminator(fake_data)
        gen_loss=generator_loss(fake_predict)
        tf.config.run_functions_eagerly(False)
        reco_loss_val = reco_loss(fake_data,detectors_rub,dt_bunlde_mask,learning_rate_fn,norm_params)
        time_loss_val = loss_time(fake_data)
        tf.config.run_functions_eagerly(True)
        gen_loss+= reco_loss_val + time_loss_val
    g_grad=gr.gradient(gen_loss,generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(g_grad, generator.trainable_variables))
    return (gen_loss,disc_loss,time_loss_val,reco_loss_val)