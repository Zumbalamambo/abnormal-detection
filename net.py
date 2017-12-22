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

class CustomOneClass(Layer):

    def __init__(self, **kwargs):
        super(CustomOneClass, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[1]), initializer='uniform', trainable=True)
        super(CustomOneClass, self).build(input_shape)

    def call(sell, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],1)

def DOC():
    inp = Input(shape=(32,32,3))
    conv1 = Conv2D(filters=32, kernel_size=(5,5))(inp)
    relu1 = Activation('relu')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(relu1)
    conv2 = Conv2D(filters=64, kernel_size=(5,5))(pool1)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    flat1 = Flatten()(pool2)
    dense1 = Dense(units=256, activation='relu')(flat1)
    drop1 = Dropout(rate=0.5)(dense1)
    doc1 = Dense(units=1, activation='relu', )
    model = Model(inputs=inp, outputs=doc1)
    return model

if __name__ == '__main__':
    model = DOC()
    model.summary()


