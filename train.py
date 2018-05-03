from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.layers import ZeroPadding2D, Input, Flatten, PReLU
# from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler,TensorBoard
import math
import os
from tqdm import tqdm
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
train = './train'

img_size = 229
batch_size = 16
epoch = 200
# x_test = []


label_map = {"Black-grass": 0, "Charlock": 1, "Cleavers": 2, "Common Chickweed": 3,"Common wheat": 4,
             "Fat Hen": 5, "Loose Silky-bent": 6, "Maize": 7, "Scentless Mayweed": 8,"Shepherds Purse": 9,
             "Small-flowered Cranesbill": 10, "Sugar beet": 11}


def img_target(path):
    image = []
    target = []

    dirs = os.listdir(path)
    for k in tqdm(range(len(dirs))):
        files = os.listdir('train/{}'.format(dirs[k]))
        for file in files:
            img = cv2.imread('train/{}/{}'.format(dirs[k], file))
            targets = np.zeros(12)
            targets[label_map[dirs[k]]] = 1

            image.append(cv2.resize(img, (img_size, img_size)))
            target.append(targets)

    image = np.array(image, np.float32) / 255
    target = np.array(target, np.uint8)
    return image, target


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
    sgd = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


def xce_model():
    model = Xception(include_top=False, weights='imagenet', input_shape=(img_size, img_size,3))
    x = model.output
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(12, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)
    return model


def learn_rate(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def train_model(image, target):
    weight = './weights.{epoch:02d}.hdf5'
    best_model = ModelCheckpoint(weight, monitor='val_acc', save_best_only=True, verbose=1)
    lr = LearningRateScheduler(learn_rate, verbose=1)
    log = TensorBoard(log_dir='./logs')
    # es = EarlyStopping(monitor='val_loss', min_delta=0, verbose=1, mode='auto')
    # model = get_model()
    model = xce_model()
    datagen = ImageDataGenerator(rotation_range=360., width_shift_range=0.3, height_shift_range=0.3,
                                 zoom_range=0.3, horizontal_flip=True, vertical_flip=True)

    x_train, x_val, y_train, y_val = train_test_split(image, target, shuffle=True, train_size=0.8)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train)//batch_size,
                        epochs=epoch, verbose=2,  validation_data=(x_val, y_val), callbacks=[best_model, lr, log])


image, target = img_target(train)

train_model(image, target)
