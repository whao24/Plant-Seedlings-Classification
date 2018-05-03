from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.layers import ZeroPadding2D, Input, Flatten, PReLU
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import *
import os
from tqdm import tqdm
import numpy as np
import cv2

test = './test1'
img_size=128
labels = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed","Common wheat",
          "Fat Hen", "Loose Silky-bent", "Maize", "Scentless Mayweed","Shepherds Purse",
          "Small-flowered Cranesbill", "Sugar beet"]


def dense_net(in_layer, n, activation, drop_rate=0.):
    dn = Dense(n)(in_layer)
    dp = Dropout(drop_rate)(dn)
    bn = BatchNormalization()(dp)
    act = Activation(activation=activation)(bn)
    return act


def conv_net(feature_batch, filters, kernel_size=(3,3),strides=(1,1),zp_flag=False):
    if zp_flag:
        zp = ZeroPadding2D((1,1))(feature_batch)
    else:
        zp = feature_batch
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides)(zp)
    bn = BatchNormalization()(conv)
    act = PReLU()(bn)
    return act


def get_model():
    in_img = Input(shape=(img_size, img_size, 3))
    conv1 = conv_net(in_img, 64, zp_flag=False)
    conv2 = conv_net(conv1, 64, zp_flag=False)
    mp1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv2)

    conv3 = conv_net(mp1, 128, zp_flag=False)
    conv4 = conv_net(conv3, 128, zp_flag=False)
    mp2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv4)

    conv5 = conv_net(mp2, 256, zp_flag=False)
    conv6 = conv_net(conv5, 256, zp_flag=False)
    mp3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv6)

    x = Flatten()(mp3)
    x = dense_net(x, 128, activation='tanh')
    x = dense_net(x, 12, activation='softmax')

    model = Model(inputs=in_img, outputs=x)
    return model


def xce_model():
    model = Xception(include_top=False, weights='imagenet', input_shape=(img_size, img_size,3))
    x = model.output
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(12, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)
    return model


model = get_model()
# model = xce_model()
model.load_weights('weights.75.hdf5')


with open('submission_2.csv', 'w') as f:
    f.write('file,species\n')
    for file in tqdm(os.listdir(test)):
        img = cv2.imread(os.path.join(test, file))

        img = cv2.resize(img, (img_size, img_size))
        image = np.array(img, np.float32) / 255
        x = np.expand_dims(image, axis=0)
        pre = model.predict(x)[0]
        f.write('{},{}\n'.format(file, labels[np.argmax(pre)]))
    print('=========Over=========')
