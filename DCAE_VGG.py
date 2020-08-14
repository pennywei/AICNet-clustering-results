import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import numpy as np
import h5py
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.layers import Input, Dense


train_dataset = h5py.File('data4000-4.h5', 'r')
x_train = np.array(train_dataset['x_train'][:])
y_train = np.array(train_dataset['y_train'][:]).astype('int')
x_test = np.array(train_dataset['x_test'][:])
y_test = np.array(train_dataset['y_test'][:]).astype('int')
x_train = np.expand_dims(x_train, -1) / 255
x_test = np.expand_dims(x_test, -1) / 255
x_train= x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


nb_epoch = 250
batch_size = 256
latent_dim = 10
img_shape = 4096


input_img = Input(shape=(4096,))
x = layers.Reshape((64, 64, 1))(input_img)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

shape_before_flattening = K.int_shape(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(10, activation='linear')(x)


'''构建AEVGG16解码网络'''
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(np.prod(shape_before_flattening[1:]),activation='relu')(x)
x = layers.Reshape(shape_before_flattening[1:])(x)
x = layers.Conv2DTranspose(512, 3, padding='same', activation='relu',strides=(2, 2))(x)
x = layers.Conv2DTranspose(256, 3, padding='same', activation='relu',strides=(2, 2))(x)
x = layers.Conv2DTranspose(128, 3, padding='same', activation='relu',strides=(2, 2))(x)
x = layers.Conv2DTranspose(64, 3, padding='same', activation='relu',strides=(2, 2))(x)
x = layers.Conv2DTranspose(64, 3, padding='same', activation='relu',strides=(2, 2))(x)
x = layers.Conv2D(1, 3, padding='same', activation='relu')(x)
x = layers.Flatten()(x)
decoded= layers.Dense(4096, activation='relu')(x)


AEVGG16 = Model(input_img, decoded)
AEVGG16.compile(optimizer='sgd', loss='mse')
AEVGG16.summary()

hist = AEVGG16.fit(x=x_train, y=x_train, shuffle=True, epochs=nb_epoch, batch_size=batch_size, validation_data=(x_test, x_test))
plot_model(AEVGG16, to_file='AEVGG16MODEL.png', show_shapes='True')




def training_vis(hist):
    loss = hist.history['loss'][1:]
    val_loss = hist.history['val_loss'][1:]

    # make a figure
    fig = plt.figure(figsize=(8,4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss, label='train_loss')
    ax1.plot(val_loss, label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('AEVGG16 archot-raddra Loss on Training and Validation Data')
    ax1.legend()

    plt.tight_layout()
    plt.show()

training_vis(hist)


aeVGG16inout4096_data100064y = AEVGG16.to_json()
with open("aeVGG16inout4096_data4000-4.json","w") as file:
    file.write(aeVGG16inout4096_data100064y)

AEVGG16.save_weights('aeVGG16inout4096_data4000-4_weights.h5')
