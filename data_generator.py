
import cv2
import numpy as np
from os import mkdir, listdir, remove, rename
from os.path import isfile, exists, splitext
import sys
from utils import *

class DataGenerator(object):
    'Generates data for Keras'

    def __init__(self, batch_size = 1, shuffle = True):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, labels, list_IDs):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)

            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

                # Generate data
                X, y = self.__data_generation(labels, list_IDs_temp)
                print (X.shape)
                print(y.shape)
                print('----------------------')

                yield X, y

    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, labels, list_IDs_temp):
        'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 32, 32, 3))
        y = np.empty((self.batch_size, 1)) #Dummy y, not used in real

        for i, ID in enumerate(list_IDs_temp):
            im_x = cv2.imread(ID, cv2.IMREAD_ANYCOLOR)
            X[i] = im_x

            # Store class
            y[i] = 1
        return X, y
