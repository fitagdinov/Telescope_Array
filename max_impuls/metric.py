import tensorflow as tf
import numpy as np
import os
import pandas as pd
import tqdm
import h5py
import argparse
def return_data():
    with h5py.File("mc_wfmax_norm.h5",'r') as f:
        norm_param=f['norm_param'][:]
        test=f['test'][:]
        train=f['train'][:472320]
        
    train=(train+1)/2
    test=(test+1)/2
    
    train_huge=train[train.max(axis=1).max(axis=1)>0.2]
    train.shape
    train=train[:2425*64]

    train_small=train[train.max(axis=1).max(axis=1)<0.2][:(25000//64)*64+2]
    train=np.append(train_small,train_huge,axis=0)
    np.random.shuffle(train)
    
    test_huge=test[test.max(axis=1).max(axis=1)>0.2]
    test_small=test[test.max(axis=1).max(axis=1)<0.2][:(8000//64)*64+2]
    test=np.append(test_small,test_huge,axis=0)
    np.random.seed(42)
    np.random.shuffle(test)
    return (train,test[:4000])

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
        noise_after,noise_befor=find_noise(generator,discriminator,image,noise_dim=noise_dim)
        fake=generator(noise_after)
        L_R=Residual_loss(image,fake)
        L_D=Discriminator_loss(image, fake,discriminator)*k_rd
        loss=L_R+L_D
        m=np.append(m,loss)
    return m.mean()



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='validation')
    parser.add_argument('-disc','--discriminator', type=str, help='dir where discriminator')
    parser.add_argument('-gen','--generators', type=str, help='dir where all generators')
    parser.add_argument('-ep','--epoch_step', type=int,default=3, help='step epochs for check generator')
    parser.add_argument('-noise','--noise_demention', type=int, help='noise demention for generator')
    parser.add_argument('-csv','--save', type=str, help='csv file to save models')
    args = parser.parse_args()
    
    train,test= return_data()
    discriminator=tf.keras.models.load_model(args.discriminator)
    df=pd.DataFrame(columns=['name','metric'])
    noise_dim=args.noise_demention
    num_ep=len(os.listdir(args.generators))
    print("____start____")
    for ep in tqdm.tqdm(range(num_ep-1,-1,-args.epoch_step),'calculate metrics'):
        try:
            generator=tf.keras.models.load_model(os.path.join(args.generators,'ep'+str(ep)))
            m=metric(generator,discriminator,test,noise_dim=noise_dim,batch=32,name='ep'+str(ep))
        except OSError:
            m="Not found model"
        df_new=pd.DataFrame([['epoch_'+str(ep),m]],columns=['name','metric'])
        df=pd.concat([df,df_new])
    df.to_csv(args.save)
    print('____finish____')
