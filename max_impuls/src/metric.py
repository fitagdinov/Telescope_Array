import tensorflow as tf
import numpy as np
import os
import pandas as pd
import tqdm
import h5py
import argparse

np.random.seed=42
file_name='../../mc_hadrons_qgs34_0010.h5'
data_name=['pr-q4-9yr']
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
#     np.random.shuffle(train)
#     train.shape
    return (train_huge,test[:num_test])

def Residual_loss(data_true, data_fake):
    return tf.reduce_mean( tf.math.square(data_true - data_fake),axis=(1,2))
def Discriminator_loss(data_true, data_fake,discriminator):
    real_pred=discriminator(data_true)
    fake_pred=discriminator(data_fake)
    return tf.math.abs( real_pred - fake_pred)

def find_noise(generator,discriminator,image,noise_dim,alpha=1.2,k_rd=0.0001):
    loss_befor=1000000000
    k=0
    optimize=tf.keras.optimizers.Adam()
    noise = tf.random.normal(shape=(image.shape[0],noise_dim))
    noise_befor=tf.identity(noise)
    list_R=[]
    list_D=[]
    for i in range(3000):
        with tf.GradientTape() as tape:
            tape.watch(noise)
            fake=generator(noise)
            loss=Residual_loss(image, fake)
            loss_disc=Discriminator_loss(image, fake,discriminator)*k_rd
            list_R.append(loss)
            list_D.append(loss_disc)
            Loss=tf.reshape(loss,(-1,1))+loss_disc
        g_grad=tape.gradient(Loss,noise)
#         optimize.apply_gradients(zip(g_grad, noise))
        noise=noise-alpha*g_grad
        if tf.math.reduce_mean((loss_befor-Loss))<0.005*tf.math.reduce_mean(Loss):
            k+=1
            if k>7:
                break
        else:
            loss_befor=Loss
            k=0
            
    return (noise,noise_befor,Loss,list_R,list_D)
def func_chunks_generators(lst, n):
        l=[]
        for i in range(0, len(lst), n):
             l.append(lst[i : i + n])
        return(l)
def metric(generator,discriminator,data,noise_dim,batch=64,k_rd=0.0001,name='???'):
    m=np.array([])
    data_list=func_chunks_generators(data,batch)
    for image in tqdm.tqdm(data_list,name):
#         print(np.array(image).shape)
        image=np.reshape(np.array(image),(-1,128,2))
        image=np.array(image)
        noise_after,noise_befor,Loss,list_R,list_D=find_noise(generator,discriminator,image,noise_dim=noise_dim)
        fake=generator(noise_after)
        L_R=Residual_loss(image,fake)
        L_D=Discriminator_loss(image, fake,discriminator)*k_rd
        loss=L_R+L_D
        m=np.append(m,loss)
    return m.mean()



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='validation')
    parser.add_argument('-md','--main_dir',type=str,help='main dir with model')
    parser.add_argument('-disc','--discriminator', type=str, help='dir where discriminator')
    parser.add_argument('-gen','--generators', type=str, help='dir where all generators')
    parser.add_argument('-ep','--epoch_step', type=int,default=3, help='step epochs for check generator')
    parser.add_argument('-noise','--noise_demention', type=int,default=398, help='noise demention for generator')
    parser.add_argument('-csv','--save', type=str, help='csv file to save models')
    args = parser.parse_args()
    
    train,test= get_data()
    if args.main_dir:
        main_dir=args.main_dir
        discriminator=os.path.join(main_dir,'save_model/discriminator/ep20')
        discriminator=tf.keras.models.load_model(discriminator)
        generator=os.path.join(main_dir,'save_model/generator')
        save=os.path.join(main_dir,'metric_for_epochs.csv')
    else:
        discriminator=tf.keras.models.load_model(args.discriminator)
        generator=args.generators
        save=args.save
    df=pd.DataFrame(columns=['name','metric'])
    noise_dim=args.noise_demention
    num_ep=len(os.listdir(generator))
    print("____start____")
    for ep in tqdm.tqdm(range(num_ep-1,-1,-args.epoch_step),'calculate metrics'):
        print('epoch is ',ep)
        try:
            generator=tf.keras.models.load_model(os.path.join(generator,'ep'+str(ep)))
            m=metric(generator,discriminator,test,noise_dim=noise_dim,batch=32,name='ep'+str(ep))
        except OSError:
            m="Not found model"
        df_new=pd.DataFrame([['epoch_'+str(ep),m]],columns=['name','metric'])
        df=pd.concat([df,df_new])
    df.to_csv(save)
    print('____finish____')
