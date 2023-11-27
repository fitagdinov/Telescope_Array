import argparse
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
import os
import pandas as pd


file_name='../../mc_hadrons_qgs34_0010.h5'
data_name=['pr-q4-9yr','pr-q3-9yr']
n_test=0.8
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)   
#dir for save  
def train_small(train,weights,threshold=0.2,step=0.05):
    intervals=np.arange(0,threshold+step,step)
    train_small_list=[]
    for train_one in tqdm.tqdm(train):
        for i in range(len(intervals)-1):
            if intervals[i]<train_one.max()<=intervals[i+1]:
                if random.random()<1/weights[i]:
                    train_small_list.append(train_one)
    return train_small_list

def get_data(num_test=2000):
    train=np.zeros((0,128,2))
    test=np.zeros((0,128,2))
    with h5py.File(file_name,'r') as f:
        for name in data_name:
            data=f[name]['wf_max'].value
            n=int(data.shape[0]*n_test)
            train=np.concatenate([train,data[:n]],axis=0)
            test=np.concatenate([test,data[n:]],axis=0)
    data=(data-data.min())/data.max()
    train=data[:n]
    test=data[n:]
    np.random.shuffle(train)
    np.random.shuffle(test)
    train_huge=train[train.max(axis=1).max(axis=1)>0.2]
#     train_small_list=train_small(train,weights=np.array([1.12229e+05, 1.12466e+05, 5.40300e+04, 3.83330e+04])/ 2.94010e+04,threshold=0.2,step=0.05)   
#     train=np.append(train_small_list,train_huge,axis=0)
#     train=train_huge
    return (train_huge,test[:num_test])
def discriminator_loss(real_output, fake_output):
    real_loss = -tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

def gradient_penalti(batch,real_data,fake_data,discriminator):
    epsilon=tf.random.uniform(shape=(batch,1,1),dtype=tf.dtypes.float32)
    interpolated=real_data-epsilon*(real_data-fake_data)# вычисление x^ как в статье
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred=discriminator(interpolated,training=True)  # D(x^)
    grads = gp_tape.gradient(pred, [interpolated])[0]# because list
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp    

@tf.function
def train_step_WGAN(labda,batch,real_data,weight_gp,weight_corr,generator,discriminator):
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
            gp=gradient_penalti(batch,real_data,fake_data,discriminator)
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
    
    
    
    
    g_grad=gr.gradient(gen_loss,generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(g_grad, generator.trainable_variables))
    return (gen_loss,disc_loss)
def image(n,dir_name,generator,num=10):
    fig, axes =plt.subplots(num,2,figsize=(50,50))
    

    for i in range(num):
        noise = tf.random.normal(shape=(1,noise_dim))
        axes[i,0].plot((generator(noise))[0,:,1],'r')#change
        axes[i,0].plot((generator(noise))[0,:,0],'b')
        j=random.randint(0,len(train))
        axes[i,1].plot(train[j,:,1],'r')
        axes[i,1].plot(train[j,:,0],'b')
        plt.suptitle("epoch:"+str(n))
        plt.savefig("{}/save_images/epoch{}.png".format(dir_name,n))
def func_chunks_generators(lst, n):
    '''передается масив и число.масив разбивается на масивы длиной не более n
    пример func_chunks_generators([1,2,3,4,5], 3) -> [[1,2,3],[4,5]]
    lst- масив
    n- число, пределяющее максимальную длину'''
    l=[]
    for i in range(0, len(lst), n):
         l.append(lst[i : i + n])
    return(l)      
def train_WGAN(epochs,train_data,batch,dir_name,generator,discriminator):
    plt.figure()
    train_data=func_chunks_generators(train_data, batch)
    for j in tqdm.tqdm(range(ep_start,epochs),'train model. Now ep num:'):
        for num in range (0,len(train_data)):
            step_data=train_data[num]   
            g,d=train_step_WGAN(labda=5,batch=len(step_data),real_data=step_data,weight_gp=10,weight_corr=1,generator=generator,discriminator=discriminator)
            d_list.append(d)
            g_list.append(g)
#             disc_loss_list.append(disc_loss)
#             gen_loss_list.append(gen_loss)
            #corr_loss_list1.append(corr_loss)
#         generator.save("del")
        #if (j % 10 == 0):
        image(j,dir_name,generator)
        discriminator.save("{}/save_model/discriminator/ep{}".format(dir_name,j))
        generator.save("{}/save_model/generator/ep{}".format(dir_name,j))
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-md','--main_dir',type=str,help='main dir with model')
    parser.add_argument('-disc','--discriminator', type=str, help='dir where discriminator')
    parser.add_argument('-gen','--generator', type=str, help='dir where generator')
    parser.add_argument('-noise','--noise_demention', type=int, help='noise demention for generator')
    parser.add_argument('-dir','--dir_save', type=str, help='dir to save models')
    parser.add_argument('-ep','--epochs', type=int,default=30, help='number of epochs')
    
    args = parser.parse_args()
    
    
    
    train,test= get_data()
    if args.main_dir:
        main_dir=args.main_dir
        discriminator=os.path.join(main_dir,'save_model/discriminator/ep0')
        discriminator=tf.keras.models.load_model(discriminator)
        generator=os.path.join(main_dir,'save_model/generator/ep0')
        generator=tf.keras.models.load_model(generator)
    else:
        generator=tf.keras.models.load_model(args.generator)
        discriminator=tf.keras.models.load_model(args.discriminator)
    #HYPER PARAMETERS
    noise_dim=args.noise_demention
    shape_data=train[0].shape
    shape=(128,2)
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    scal_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
    
    dir_name=args.dir_save
    if not (os.path.exists(dir_name)):
        os.mkdir(dir_name)
        os.mkdir('{}/save_images'.format(dir_name))
        os.mkdir('{}/save_model'.format(dir_name))
    batch=128
    epochs=args.epochs
    ep_start=0
    g_list=[]
    d_list=[]
    print('start train')
    print(gpus)
    train_WGAN(epochs,train,batch,dir_name,generator,discriminator)