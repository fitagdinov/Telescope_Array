import tensorflow as tf
shape=(6, 6, 4)
def Discriminator_model(num=''):
    input_tensor=tf.keras.Input(shape=shape)
    x= input_tensor
    x=tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1), padding='same',name="first")(x)
    x=tf.keras.activations.gelu(x)
    
    x=tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding='same',name="second")(x)
    x=tf.keras.activations.gelu(x)
#     x=tf.keras.layers.AveragePooling2D((2,2),padding='same')(x)
    
    
    # ПОПРОБОВАТЬ Ч1
    
    # mask=tf.keras.layers.MaxPool2D(2)(mask)
    
    
    x=tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1), padding='same',name="therd")(x)
    x=tf.keras.activations.gelu(x)
#     x=tf.keras.layers.AveragePooling2D((2,2),padding='same')(x)
    
    
    x=tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(1,1), padding='same',name="forth")(x)
#     x=tf.keras.layers.LeakyReLU()(x)
    x=tf.keras.activations.gelu(x)
    
    x=tf.keras.layers.Conv2D(filters=128, kernel_size=(2,2), strides=(1,1), padding='same',name="fifth")(x)
#     x=tf.keras.layers.LeakyReLU()(x)
    x=tf.keras.activations.gelu(x)
    # Ч2
#     x=x*mask
    
    x=tf.keras.layers.Flatten()(x)
#     x=tf.keras.layers.Dropout(rate=0.1)(x)

    x=tf.keras.layers.Dense(units=30)(x)
    x=tf.keras.activations.gelu(x)
    x=tf.keras.layers.Dense(units=1)(x)
    
    model= tf.keras.Model(input_tensor,x,name="Discriminator_model_num_{}".format(num))
    return model
