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
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



for noise_dim in range(140,200,20):
    with h5py.File("mc_wfmax_norm.h5",'r') as f:
        norm_param=f['norm_param'][:]
        test=f['test'][:]
        train=f['train'][:472320]
    # mean=train.mean()
    # train=train-mean
    # train=train/(np.absolute(train)).max()
    train=(train+1)/2
    test=(test+1)/2


    shape_data=train[0].shape
    shape=(128,2)
    # generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0, beta_2=0.9)
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    scal_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)


    def Discriminator_model(num=''):
        input_tensor=tf.keras.Input(shape=(128,2))
        x= input_tensor

        x=tf.keras.layers.ZeroPadding1D(10)(x)
    #     x=tf.keras.layers.Conv1D(filters=32, kernel_size=(5), strides=(1), padding='same')(x)
    #     x=tf.keras.layers.LeakyReLU()(x)

        x=tf.keras.layers.Conv1D(filters=32, kernel_size=(5), strides=(1), padding='same',name="first")(x)
        x=tf.keras.layers.MaxPooling1D(2,padding='same')(x)
        x=tf.keras.layers.Dropout(rate=0.2)(x)
        x=tf.keras.layers.LeakyReLU()(x)
        x=tf.keras.layers.Conv1D(filters=64, kernel_size=(5), strides=(1), padding='same')(x)
        x=tf.keras.layers.MaxPooling1D(2,padding='same')(x)
        x=tf.keras.layers.Dropout(rate=0.2)(x)
        x=tf.keras.layers.LeakyReLU()(x)

        x=tf.keras.layers.Conv1D(filters=64, kernel_size=(5), strides=(1), padding='same')(x)
        x=tf.keras.layers.MaxPooling1D(2,padding='same')(x)
        x=tf.keras.layers.Dropout(rate=0.2)(x)
        x=tf.keras.layers.LeakyReLU()(x)

        x=tf.keras.layers.Conv1D(filters=64, kernel_size=(5), strides=(1), padding='same')(x)# kern=4
        x=tf.keras.layers.MaxPooling1D(2,padding='same')(x)
        x=tf.keras.layers.Dropout(rate=0.2)(x)
        x=tf.keras.layers.LeakyReLU()(x)

        x=tf.keras.layers.Conv1D(filters=128, kernel_size=(5), strides=(1), padding='same')(x)# kern=4
        x=tf.keras.layers.MaxPooling1D(2,padding='same')(x)
        x=tf.keras.layers.Dropout(rate=0.2)(x)
        x=tf.keras.layers.LeakyReLU()(x)

        x=tf.keras.layers.Flatten()(x)
        x=tf.keras.layers.Dropout(rate=0.1)(x)

        x=tf.keras.layers.Dense(units=100,activation='relu')(x)
        x=tf.keras.layers.Dense(units=1)(x)

        model= tf.keras.Model(input_tensor,x,name="Discriminator_model_num_{}".format(num))
        return model



    def Generator_model(num="",noise_dim=200): 
        input_tensor=tf.keras.Input(shape=(noise_dim,))
        x=tf.keras.layers.Dense(units=200,use_bias=False)(input_tensor)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.LeakyReLU()(x)

        x=tf.keras.layers.Dense(units=33*6,use_bias=False)(x)
        x=tf.keras.layers.Dropout(rate=0.1)(x)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.LeakyReLU()(x)

        x=tf.keras.layers.Reshape((33,6,1))(x)

        x=tf.keras.layers.Conv2DTranspose(128, (5,4), strides=(2,1), padding='same', use_bias=False,data_format='channels_last')(x)
        x=tf.keras.layers.Dropout(rate=0.1)(x)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.LeakyReLU()(x)

        x=tf.keras.layers.Conv2DTranspose(64, (5,2), strides=(2,1), padding='same', use_bias=False,data_format='channels_last')(x)
        x=tf.keras.layers.Dropout(rate=0.1)(x)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.LeakyReLU()(x)
    #     print(x.shape)
        x=tf.keras.layers.Conv2DTranspose(32, (5,2), strides=(2,1), padding='same', use_bias=False,data_format='channels_last')(x)
        x=tf.keras.layers.Dropout(rate=0.1)(x)

        x=tf.keras.layers.AveragePooling2D((2,1),padding='same')(x)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.LeakyReLU()(x)

    #     x=tf.keras.layers.Conv2DTranspose(16, (5,1), strides=(2,1), padding='same', use_bias=False,data_format='channels_last')(x)
    #     x=tf.keras.layers.Dropout(rate=0.1)(x)
    #     x=tf.keras.layers.BatchNormalization()(x)
    #     x=tf.keras.layers.LeakyReLU()(x)

        x=tf.keras.layers.Conv2DTranspose(8, (5,2), strides=(2,1), padding='same', use_bias=False,data_format='channels_last')(x)
        x=tf.keras.layers.Dropout(rate=0.1)(x)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.LeakyReLU()(x)
        x=tf.keras.layers.AveragePooling2D((2,1),padding='same')(x)

    #     print(x.shape)
        x=tf.keras.layers.Conv2D(1, (5,2), strides=(1,1), padding='same', use_bias=False,data_format='channels_last')(x)
        x=tf.keras.layers.Dropout(rate=0.1)(x)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.activations.sigmoid(x)
        x=tf.keras.layers.Cropping2D((2,2))(x) #change
        x=tf.keras.layers.Reshape((128,2))(x)

        model= tf.keras.Model(input_tensor,x,name="Generator_model_{}".format(num))
    #     assert model.output_shape == (None, 128, 2, 1)
        return model





    def discriminator_loss(real_output, fake_output):
    #     real_loss = loss_function(tf.ones_like(real_output), real_output)
    #     fake_loss = loss_function(tf.zeros_like(fake_output), fake_output)
        real_loss = -tf.reduce_mean(real_output)
        fake_loss = tf.reduce_mean(fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
    #     return loss_function(tf.ones_like(fake_output), fake_output)
        return -tf.reduce_mean(fake_output)

    def gradient_penalti(batch,real_data,fake_data):
    #   alpha = tf.random.normal([batch, 1], 0.0, 1.0)
    #   diff = fake_data - real_data
    #   interpolated = real_data + alpha * diff
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
    def train_step_WGAN(labda,batch,real_data,weight_gp,weight_corr,noise_dim):
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




        g_grad=gr.gradient(gen_loss,generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(g_grad, generator.trainable_variables))
        return (gen_loss,disc_loss)




    def image(n,noise_dim,num=10):
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
        plt.close()



    def train_WGAN(epochs,train_data,batch,noise_dim):
        plt.figure()
        for j in range(ep_start,epochs):
            for num in range (0,len(train_data),batch):
                step_data=train_data[num:num+batch]
                g,d=train_step_WGAN(labda=5,batch=batch,real_data=step_data,weight_gp=10,weight_corr=1,noise_dim=noise_dim)
                d_list.append(d)
                g_list.append(g)
    #             disc_loss_list.append(disc_loss)
    #             gen_loss_list.append(gen_loss)
                #corr_loss_list1.append(corr_loss)
    #         generator.save("del")
    #         if (j % 10 == 0 or j <25):
            image(j,noise_dim=noise_dim)
            discriminator.save("{}/save_model/discriminator/ep{}".format(dir_name,j))
            generator.save("{}/save_model/generator/ep{}".format(dir_name,j))

    batch=64
    epochs=50
    ep_start=0
    g_list=[]
    d_list=[]
    dir_name="noise/{}".format(str(noise_dim))
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        os.mkdir('{}/save_images'.format(dir_name))
        os.mkdir('{}/save_model'.format(dir_name))
    print("start:",noise_dim)
    generator=Generator_model(noise_dim=noise_dim)
    discriminator=Discriminator_model()
    train_WGAN(epochs,train,batch,noise_dim)
