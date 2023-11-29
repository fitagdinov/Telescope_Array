import tensorflow as tf
import numpy as np
import os
import pandas as pd
import tqdm
import h5py
import metric
import argparse

np.random.seed=42
file_name='../../mc_hadrons_qgs34_0010.h5'
trainable=['pr-q4-9yr']
# not_trainable=['pr-q3-9yr']
not_trainable=['sddata_9yr']
print('trainable',trainable)
print('not_trainable',not_trainable)
n_test=0.8
def train_small(train,weights,threshold=0.2,step=0.05):
    intervals=np.arange(0,threshold+step,step)
    train_small_list=[]
    for train_one in tqdm.tqdm(train):
        for i in range(len(intervals)-1):
            if intervals[i]<train_one.max()<=intervals[i+1]:
                if random.random()<1/weights[i]:
                    train_small_list.append(train_one)
    return train_small_list

def get_data(data_name,num_test=2000):
    train=np.zeros((0,128,2))
    test=np.zeros((0,128,2))
    with h5py.File(file_name,'r') as f:
        for name in data_name:
            data=f[name]['wf_max'].value
            data=(data-data.min())/data.max()
            n=int(data.shape[0]*n_test)
            train=np.concatenate([train,data[:n]],axis=0)
            test=np.concatenate([test,data[n:]],axis=0)
    np.random.shuffle(train)
    np.random.shuffle(test)
    train_huge=train[train.max(axis=1).max(axis=1)>0.2]
#     train_small_list=train_small(train,weights=np.array([1.12229e+05, 1.12466e+05, 5.40300e+04, 3.83330e+04])/ 2.94010e+04,threshold=0.2,step=0.05)   
#     train=np.append(train_small_list,train_huge,axis=0)
#     train=train_huge
#     np.random.shuffle(train)
#     train.shape
    return (train_huge,test[:num_test])
def amplitude(data):
    # shape -1 128 2
    return data.max(axis=(1,2))
def metric_main(generator,discriminator,data,noise_dim,batch=64,k_rd=0.0001,name='???'):
    m=np.array([])
    data_list=metric.func_chunks_generators(data,batch)
    ampl=np.array([])
    for image in tqdm.tqdm(data_list,name):
#         print(np.array(image).shape)
        image=np.reshape(np.array(image),(-1,128,2))
        image=np.array(image)
        noise_after,noise_befor,Loss,list_R,list_D=metric.find_noise(generator,discriminator,image,noise_dim=noise_dim)
        fake=generator(noise_after)
        L_R=tf.reshape(metric.Residual_loss(image,fake),(-1))
        L_D=tf.reshape(metric.Discriminator_loss(image, fake,discriminator)*k_rd,(-1))
        loss=L_R+L_D
        print('loss.shape',loss.shape)
        m=np.append(m,loss)
        ampl=np.append(ampl,amplitude(image))
    return m, ampl

def main(disc_path,gen_path,noise_dim,save):
    discriminator = tf.keras.models.load_model(disc_path)
    generator = tf.keras.models.load_model(gen_path)
    train = get_data(trainable)[1] # test dataset from model which is used in training
    test = get_data(not_trainable)[1] # test dataset from model which is not used in training
    df=pd.DataFrame(columns=['name','metric'])
    print("____start____")
    m_full_train,ampl_full_train=metric_main(generator,discriminator,train,noise_dim=noise_dim,batch=128,name='train')
    m_full_test,ampl_full_test=metric_main(generator,discriminator,train,noise_dim=noise_dim,batch=128,name='test')
    train_name='_'.join(trainable)
    test_name='_'.join(not_trainable)
    df = pd.DataFrame(columns=[f'train__{train_name}__mertic',f'train__{train_name}__ampl',
                             f'test__{test_name}__mertic',f'test__{test_name}__ampl'])
    print(m_full_train.shape,ampl_full_train.shape)
    print(m_full_test.shape,ampl_full_test.shape)
    df[f'train__{train_name}__mertic']=m_full_train
    df[f'train__{train_name}__ampl']=ampl_full_train
    df[f'test__{test_name}__mertic'] = m_full_test
    df[f'test__{test_name}__ampl'] = ampl_full_test
    df.to_csv(save,index=False)
    print('____finish____')


if __name__ == "__main__":
    
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    parser = argparse.ArgumentParser(description='validation')
    parser.add_argument('-disc', '--discriminator', type=str, help='dir where discriminator')
    parser.add_argument('-gen', '--generator', type=str, help='dir where all generators')
    parser.add_argument('-noise', '--noise_demention', type=int, default=398, help='noise demention for generator')
    parser.add_argument('-save', '--save', type=str, help='csv file to save models')
    args = parser.parse_args()
    disc_path=args.discriminator
    gen_path=args.generator
    noise_dim=args.noise_demention
    save=args.save
    main(disc_path, gen_path, noise_dim, save)
