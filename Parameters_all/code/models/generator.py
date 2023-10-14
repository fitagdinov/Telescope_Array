import tensorflow as tf
def Generator_model(num="",noise_dim=100): 
    input_tensor=tf.keras.Input(shape=(noise_dim,))
    
    x=tf.keras.layers.Dense(units=16*32,use_bias=False)(input_tensor)
#     x=tf.keras.layers.Dropout(rate=0.1)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.activations.gelu(x)
    x=tf.keras.layers.Reshape((4,4,32))(x)
    
    x=tf.keras.layers.Conv2D(32, (2,2), strides=(1,1), padding='same', use_bias=False,data_format='channels_last',
                            name='C_0')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.activations.gelu(x)
    
    
    
    ## 6x6
    x=tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same', use_bias=False,data_format='channels_last',
                            name='T_1')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.activations.gelu(x)
    
    # 12#12
    x=tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same', use_bias=False,data_format='channels_last',
                            name='T_2')(x)
#     x=tf.keras.layers.Dropout(rate=0.1)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.activations.gelu(x)
    
    x=tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same', use_bias=False,data_format='channels_last',
                            name='T_3')(x)
    x=tf.keras.layers.AveragePooling2D((2,2),padding='same')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.activations.gelu(x)
    
    x=tf.keras.layers.Conv2D(16, (2,2), strides=(1,1), padding='same', use_bias=False,data_format='channels_last',
                            name='c_2')(x)
#     x=tf.keras.layers.Dropout(rate=0.1)(x)
    x=tf.keras.layers.AveragePooling2D((2,2),padding='same')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.activations.gelu(x)
# #     print(x.shape)


    x=tf.keras.layers.Conv2D(4, (2,2), strides=(1,1), padding='same', use_bias=False,data_format='channels_last',
                            name='C_3')(x)
    x=tf.keras.activations.sigmoid(x)
    x=tf.keras.layers.Cropping2D((1,1))(x)

    model= tf.keras.Model(input_tensor,x,name="Generator_model_{}".format(num))
    return model