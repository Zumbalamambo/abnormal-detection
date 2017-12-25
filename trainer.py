import cv2
from keras import backend as K
import net
import numpy as np
from os import listdir
from os.path import isfile, join
import time
import utils
from data_generator import DataGenerator

def dummy_loss(y_true, y_pred):
    zero = K.variable(0.0)
    return zero

def train(batch_path):
    #Get dataset ids
    X, Y = utils.get_dataset(batch_path)

    print("-------------------------")
    model = net.DOC(training=True)
    print("Loaded model")
    print("-------------------------")
    model.compile(loss=dummy_loss, optimizer='SGD')
    print("Training")
    start = time.time()
    model.fit(X, Y, batch_size=200, epochs=3, validation_split=0.3)
    end = time.time()
    print("Training done, time= ", end-start)
    model.save_weights("doc.h5")
    print("Saved model, done")

if __name__ == '__main__':
    train('../UCSDped1/TrainBatch')



        