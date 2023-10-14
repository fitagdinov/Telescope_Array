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
from joblib import Parallel, delayed

from models import discriminator  as disc
from models import generator as gen
dir_name="../deep_8"
noise_dim=50
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)

param_names=['signal','pl_fr','real_wf','mask']



gpus = tf.config.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if os.path.exists(dir_name):
    print("\n THIS DIR ALREADY EXISTS \n ALL FILES WILL BE DELETED \n YOU REALY WANT TITH ")
#     time.sleep(10)
else:
    os.mkdir(dir_name)
    os.mkdir('{}/save_images'.format(dir_name))
    os.mkdir('{}/save_model'.format(dir_name))
    
    
    
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
        print(max_c,min_c)
        data[:,:,:,i]=(data[:,:,:,i]-min_c)/(max_c-min_c)

        norm_params[param_names[i]]=np.array([max_c,min_c])
    return norm_params
def renormin(data_not_change,norm_params,param_names=param_names,log=True):
    data=np.copy(data_not_change)
    for i in range(data.shape[-1]):
        max_c, min_c=norm_params[param_names[i]]
        data[:,:,:,i]=data[:,:,:,i]*(max_c-min_c) + min_c
        if (i==0 and log):
            data[:,:,:,i]=tf.math.exp(data[:,:,:,i])-1
        elif (i==1 or i==2):
            data[:,:,:,i]=data[:,:,:,i]*1e6
    return data
with h5py.File("../../mk_all_old.h5",'r') as f:
#     print(list(f.keys()))
    
    print(f['sddata'].keys())
    print(f['sddata']['dt'])# вырезки центрированные около середины ливня size 6x6
    data=f['sddata']['dt'][:,:,:,3:7]
    print(f['sddata']['ev_params'][:,1])
#     data=f['sddata']['wf_fl'][:]
    print(f['norm_param_global_18.0_qgs3_all']['wf_fl']['mean'][:])
norm_params=norming(data)
shape=data.shape[1:]
def im_one(i,data,axs):
    labels=data[i,:,:,0]
    sns.heatmap(data[i,:,:,2],annot=labels,ax=axs[i,0],vmin=0.0, vmax=1.0)
    sns.heatmap(data[i,:,:,2]-data[i,:,:,1],annot=data[i,:,:,3],ax=axs[i,1],vmin=-1.0, vmax=1.0)

def image_signal(data,fake=None,dir_name=None,ep='not_ep'):
    n=data.shape[0]
    if (fake is None):
        fig, axs = plt.subplots(nrows=n, ncols=2, figsize=(10,n*5))
#         Parallel(n_jobs=-1)(delayed(im_one)(i,data,axs) for i in range(n))
        for i in range(n):
            im_one(i,data,axs)
                           
        fig.suptitle('only real')
    else:
        fig, axs = plt.subplots(nrows=n, ncols=4, figsize=(10*2,n*5))
        for i in range(n):
            
            
            data_after=renormin(np.array([data[i,:,:,:]]),norm_params)
            data_after[0]=tf.where(tf.cast(tf.expand_dims(data_after[0,:,:,-1],axis=-1),tf.bool),data_after[0,:,:,:],np.nan)
            sns.heatmap(data_after[0,:,:,0],annot=data_after[0,:,:,0],ax=axs[i,2],fmt=".1f")
            sns.heatmap(data_after[0,:,:,1],annot=data_after[0,:,:,2],ax=axs[i,3],fmt=".2f")
            
            #fake
            threshold=0.5
            mask_round=tf.where(fake[i,:,:,3]>threshold,1.0,0.0)
            mask_round=tf.cast(mask_round,tf.float32)
            fake_after_mask=np.array([fake[i]])
            #renorming
            
            
            fake_after_renorming=renormin(fake_after_mask,norm_params)
            fake_after_mask=tf.where(tf.cast(tf.expand_dims(mask_round,axis=-1),tf.bool),fake_after_renorming[0,:,:,:],np.nan)# nan

            
            fake_after_mask=fake_after_mask.numpy()
            fake_after_mask[:,:,-1]=mask_round
            fake_after_mask[:,:,-1]=np.where(mask_round<0.9,np.nan,1)
            sns.heatmap(fake_after_mask[:,:,0],annot=fake_after_mask[:,:,0],ax=axs[i,0],fmt=".1f")
            
            sns.heatmap(fake_after_mask[:,:,1],annot=fake_after_mask[:,:,2],ax=axs[i,1],fmt=".2f")
        fig.suptitle('fake     /    real')
    if dir_name:
        plt.savefig("{}/save_images/epoch{}.png".format(dir_name,ep))
        
def images(generator,num=10,data=data,noise_dim=noise_dim,dir_name='',ep='not_ep'):
    rand=np.random.choice(np.arange(data.shape[0]),num)
    data_for_plot=np.zeros((num,shape[0],shape[1],shape[2]))
    for i in range(num):
        data_for_plot[i]=data[rand[i]]
    noise = tf.random.normal(shape=(num,noise_dim))
    fake=generator(noise)
    image_signal(data_for_plot,fake=fake,dir_name=dir_name,ep=ep)
discriminator=disc.Discriminator_model()
generator=gen.Generator_model()
def discriminator_loss(real_output, fake_output):
#     real_loss = loss_function(tf.ones_like(real_output), real_output)
#     fake_loss = loss_function(tf.zeros_like(fake_output), fake_output)
    real_loss = -tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

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

def gradient_penalti(batch,real_data,fake_data):
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
@tf.function
def train_step_WGAN(labda,batch,real_data,weight_gp,weight_corr):
  # labda --> number learling critic
  #weight --> weight gradient_penalti

  #learning critic
    for i in range(labda):
        with tf.GradientTape() as gr:
            noise = tf.random.normal(shape=(batch,noise_dim))
            fake_data=generator(noise)#change
#             print('f')
            real_data=real_data
            fake_predict=discriminator(fake_data)
            real_predict=discriminator(real_data)
            real_data=tf.cast(real_data,dtype=tf.float32)
            gp=gradient_penalti(batch,real_data,fake_data)
            disc_loss=discriminator_loss(real_predict,fake_predict)+weight_gp*gp # critic loss include GP
        d_grad=gr.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(d_grad, discriminator.trainable_variables))


  #learning generator
    noise = tf.random.normal(shape=(batch,noise_dim))
    with tf.GradientTape() as gr:
        fake_data=generator(noise)
        fake_predict=discriminator(fake_data)
#         corr_loss=correletion_loss(fake_data)
        gen_loss=generator_loss(fake_predict)# +weight_corr*corr_loss
        gen_loss=gen_loss+ loss_time(fake_data)
    
    
    
    
    g_grad=gr.gradient(gen_loss,generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(g_grad, generator.trainable_variables))
    return (gen_loss,disc_loss)
def func_chunks_generators(lst, n):
    '''передается масив и число.масив разбивается на масивы длиной не более n
    пример func_chunks_generators([1,2,3,4,5], 3) -> [[1,2,3],[4,5]]
    lst- масив
    n- число, пределяющее максимальную длину'''
    l=[]
    for i in range(0, len(lst), n):
         l.append(lst[i : i + n])
    return(l)
batch=64
epochs=200
ep_start=0
g_list=[]
d_list=[]
def train_WGAN(epochs,train_data,batch):
    plt.figure()
    train_data=func_chunks_generators(train_data, batch)
    for j in tqdm_notebook(range(ep_start,epochs),'ep'):
        for num in range(0,len(train_data)):
            step_data=train_data[num]
            g,d=train_step_WGAN(labda=5,batch=len(step_data),real_data=step_data,weight_gp=10,weight_corr=1)
            d_list.append(d)
            g_list.append(g)
#         if (j % 10 == 0):
        
        images(generator,num=10,noise_dim=noise_dim,dir_name=dir_name,ep=j)
        discriminator.save("{}/save_model/discriminator/ep{}".format(dir_name,j))
        generator.save("{}/save_model/generator/ep{}".format(dir_name,j))
    plt.close()

train_WGAN(epochs,data,batch)
plt.close()
plt.figure()
plt.plot(d_list[5000::100],'r')
plt.plot(g_list[5000::100],'g')
plt.legend(['disc','gen'])
plt.title(dir_name)
plt.savefig(os.path.join(dir_name,'loss.jpg'))