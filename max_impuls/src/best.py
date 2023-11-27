import tensorflow as tf
import numpy as np
import os
import pandas as pd
import tqdm
import h5py
import argparse
import glob
from metric import *
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
def find_generator(path:str):
    csv_name=glob.glob(path+'/*.csv')[0]
    df=pd.read_csv(csv_name)
    epoch=list(df[df.metric==df.metric.min()]['name'])[0]
    epoch=epoch.replace('epoch_','ep')
    return os.path.join(path+'/save_model/generator',epoch)
def find_noise(path:str):
    with open (os.path.join(path,'noise_fim.txt'),'r') as file:
        noise=int(file.read())
    return noise
def metric_one_gen(path,disc_names,noise_dim,test):
    all_metrics=np.zeros(4)
    generator_name=find_generator(path)
    generator=tf.keras.models.load_model(generator_name)
    for i in range(4):
        discriminator=tf.keras.models.load_model(disc_names[i])
        try:
            m=metric(generator,discriminator,test,noise_dim=noise_dim,batch=32,name=disc_names[i])
            all_metrics[i]=m
        except OSError:
            m="Not found model"
            all_metrics[i]=np.nan
    with open (os.path.join(path,'all_metrics.txt'),'w') as file:
        file.writelines('  '.join([str(metr) for metr in all_metrics]))
    return (all_metrics,generator_name)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='validation')
    parser.add_argument('-path','--main_path', type=str, help='main_path')
    args = parser.parse_args()
    main_path=args.main_path
    train,test= return_data()
    all_path=glob.glob(main_path+'/*')
    disc_names=[i+'/save_model/discriminator/ep29' for i in all_path]
    generator_path=all_path.copy()
    mean_metric_dict={}
    print(generator_path)
    for path in generator_path:
        noise_dim=find_noise(path)
        m,generator_name=metric_one_gen(path,disc_names,noise_dim,test)
        mean_metric=m.mean()
        mean_metric_dict[generator_name]=mean_metric
    df=pd.DataFrame([mean_metric_dict])
    df.to_csv(main_path+'/mean_metric.csv')
        
        
        