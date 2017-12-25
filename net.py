import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.engine.topology import Layer
from keras import backend as K
from keras import Sequential

import numpy as np

class Sign(Layer):
    def __init__(self, **kwargs):
        super(Sign, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x):
        return K.sign(x)

    def compute_output_shape(self, input_shape):
        return input_shape

def DOC(training = False):
    
    inp = Input(shape=(32,32,3))
    ##############################################
    # CNN part
    ##############################################
    conv1 = Conv2D(filters=32, kernel_size=(5,5), name="cnn_conv1")(inp)
    relu1 = Activation('relu', name="cnn_relu1")(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2), name="cnn_pool1")(relu1)
    conv2 = Conv2D(filters=64, kernel_size=(5,5), name="cnn_conv2")(pool1)
    pool2 = MaxPool2D(pool_size=(2, 2), name="cnn_pool2")(conv2)
    flat1 = Flatten(name="cnn_flat1")(pool2)
    dense1 = Dense(units=256, activation='relu', name="cnn_dense1")(flat1)
    drop1 = Dropout(rate=0.5, name="cnn_drop1")(dense1)
    ##############################################
    # One-class part
    ##############################################
    doc1 = Dense(units=1, name="doc_doc1")(drop1)
    sign1 = Sign(name="doc_sign1")(doc1)
    model = Model(inputs=inp, outputs=sign1)
    
    '''

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), name="cnn_conv1", input_shape=(32,32,3)))
    model.add(Activation('relu', name="cnn_relu1"))
    model.add(MaxPool2D(pool_size=(2, 2), name="cnn_pool1"))
    model.add(Conv2D(filters=64, kernel_size=(5,5), name="cnn_conv2"))
    model.add(MaxPool2D(pool_size=(2, 2), name="cnn_pool2"))
    model.add(Flatten(name="cnn_flat1"))
    model.add(Dense(units=256, activation='relu', name="cnn_dense1"))
    model.add(Dropout(rate=0.5, name="cnn_drop1"))
    '''



    ###############################################
    # Add loss
    ###############################################
    if training==True:
        add_loss(model)

    return model

def add_loss(model):
    loss = K.variable(0.0)

    doc1 = model.layers[-2].output
    loss += K.relu(-doc1)

    sign1 = model.layers[-1].output
    ones = K.ones_like(sign1)
    loss += (sign1-ones)

    loss = K.sum(loss)

    layers = model.layers[:]
    temp_loss = 0.0
    for layer in layers:
        if layer.name == "doc_doc1":
            temp_loss += np.sum(np.square(np.linalg.norm(layer.get_weights()[0].flatten())))
        else:
            for weight in layer.get_weights():
                temp_loss += np.sum(np.square(np.linalg.norm(weight.flatten())))
    temp_loss = temp_loss/2
    loss += temp_loss

    loss -= (layers[-2]).get_weights()[0]
    model.layers[-1].add_loss(loss)

if __name__ == '__main__':
    model = DOC(True)
    model.summary()


