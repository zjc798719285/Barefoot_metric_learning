from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, SeparableConv2D
from keras.layers import Dense, BatchNormalization, Dropout, Activation, regularizers
from keras.layers.merge import concatenate
import keras as k
import tensorflow as tf
#一个666的模型
class my_alex(object):
    def __init__(self, x, keep_prob, num_class):
        self.x = x
        self.keep_prob = keep_prob
        self.num_class = num_class
        self.init = k.initializers.glorot_normal()
        self.regular = regularizers.l1_l2(l1=0.1, l2=0.5)
        self.trainable1 = True
        self.trainable2 = True
    def predict(self):
        net1 = Conv2D(64, (7, 7), padding='same', strides=[2, 2],
                      kernel_initializer=self.init, activation='relu', trainable=self.trainable1)(self.x)
        net1 = block(x=net1, f=64, init=self.init, trainable=self.trainable1)
        net2 = Conv2D(128, (3, 3), padding='same', strides=[1, 1], activation='relu',
                      kernel_initializer=self.init, trainable=self.trainable1)(net1)
        net2 = MaxPooling2D((3, 3), strides=(2, 2), trainable=self.trainable1)(net2)
        net2 = block(x=net2, f=128, init=self.init, trainable=self.trainable1)
        net2 = block(x=net2, f=128, init=self.init, trainable=self.trainable1)
        net3 = Conv2D(256, (3, 3), padding='same', strides=[1, 1], activation='relu',
                      kernel_initializer=self.init, trainable=self.trainable1)(net2)
        net3 = MaxPooling2D((3, 3), strides=(2, 2))(net3)
        net3 = block(x=net3, f=256, init=self.init, trainable=self.trainable2)
        net3 = block(x=net3, f=256, init=self.init, trainable=self.trainable2)
        net4 = MaxPooling2D((3, 3), strides=(2, 2))(net3)
        net4 = Conv2D(256, (3, 3), padding='same', strides=[1, 1], activation='relu',
                      kernel_initializer=self.init, trainable=self.trainable2)(net4)
        net4 = block(x=net4, f=256, init=self.init, trainable=self.trainable2)
        net5 = tf.reshape(net4, [-1, 7*2*256])
        net_m = Dense(4096, activation='tanh', trainable=self.trainable2)(net5)
        net_s = Dense(256, trainable=self.trainable2)(net_m)
        return net_s, net_m
def block(x, f, init,trainable):
    b1 = SeparableConv2D(filters=f, kernel_size=(3, 3), padding='same', depth_multiplier=1)(x)
    b1 = Conv2D(f, (1, 1), padding='same', strides=[1, 1], activation='relu',
                      kernel_initializer=init, trainable=trainable)(b1)
    b2 = SeparableConv2D(filters=f, kernel_size=(3, 3), padding='same', depth_multiplier=1)(b1)
    b2 = Conv2D(f, (1, 1), padding='same', strides=[1, 1], activation='relu',
                kernel_initializer=init, trainable=trainable)(b2)
    b3 = b2+x
    return b3
