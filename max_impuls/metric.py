import tensorflow as tf
import numpy as np
import os
import pandas as pd
import tqdm
import h5py


with h5py.File("mc_wfmax_norm.h5",'r') as f:
    print(list(f.keys()))
    norm_param=f['norm_param'][:]
    test=f['test'][:]
    train=f['train'][:472320]
    print(f['train'].shape)
# mean=train.mean()
# train=train-mean
# train=train/(np.absolute(train)).max()
train=(train+1)/2
test=(test+1)/2


noise_dim_list=np.arange(100,320,20)
list_dirs=[os.path.join('noise',str(i)) for i in noise_dim_list]

def Residual_loss(data_true, data_fake):
    return tf.reduce_mean( tf.math.abs(data_true - data_fake),axis=(1,2))
def Discriminator_loss(data_true, data_fake):
    real_pred=discriminator(data_true)
    fake_pred=discriminator(data_fake)
    return tf.math.abs( real_pred - fake_pred)
# loss_L=[]
# loss_D=[]
def find_noise(generator,discriminator,image,noise_dim,alpha=1.2,k_rd=0.0001):
#     global loss_L
    noise = tf.random.normal(shape=(len(image),noise_dim))
    noise_befor=tf.identity(noise)
    for i in range(3000):
        with tf.GradientTape() as tape:
            tape.watch(noise)
            fake=generator(noise)
            loss=Residual_loss(image, fake)
            loss_disc=Discriminator_loss(image, fake)*k_rd
#             print(loss.shape,loss_disc.shape)
            Loss=tf.reshape(loss,(-1,1))+loss_disc
#             print(Loss.shape)
#             loss_L.append(Loss)
        g_grad=tape.gradient(Loss,noise)
#         print(noise.shape,g_grad.shape)
        noise=noise-alpha*g_grad
    return (noise,noise_befor)
def func_chunks_generators(lst, n):
    '''передается масив и число.масив разбивается на масивы длиной не более n
    пример func_chunks_generators([1,2,3,4,5], 3) -> [[1,2,3],[4,5]]
    lst- масив
    n- число, пределяющее максимальную длину'''
    l=[]
    for i in range(0, len(lst), n):
         l.append(lst[i : i + n])
    return(l)
def metric(generator,discriminator,data,batch,noise_dim,k_rd=0.0001,name='???'):
    m=np.array([])
    data_list=func_chunks_generators(data,batch)
    for image in tqdm.tqdm(data_list,name):
        image=tf.reshape(image,(-1,128,2))
        noise_after,noise_befor=find_noise(generator,discriminator,image,noise_dim=noise_dim)
        fake=generator(noise_after)
        L_R=Residual_loss(image,fake)
        L_D=Discriminator_loss(image, fake)*k_rd
        loss=L_R+L_D
        m=np.append(m,loss)
    return m.mean()
df=pd.DataFrame(columns=['name','metric'])
for i in range(len(list_dirs)):
    s=list_dirs[i]
    noise_dim=noise_dim_list[i]
    try:
        generator=tf.keras.models.load_model('{}/save_model/generator/ep20'.format(s))
        discriminator=tf.keras.models.load_model('{}/save_model/discriminator/ep20'.format(s))
        m=metric(generator,discriminator,test,128,noise_dim=noise_dim,name=s)
    except OSError:
        m="Not found model"
    df_new=pd.DataFrame([[s,m]],columns=['name','metric'])
    df=pd.concat([df,df_new])
df.to_csv('metric_noise.csv')
