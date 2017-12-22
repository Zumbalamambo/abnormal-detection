import numpy as np
import time

if __name__ == '__main__':
    A = np.zeros((1000000))
    start = time.time()
    for i in range (10000):
        A[i] = i*2
    end = time.time()
    print ('time= ', end-start)

    start = time.time()
    A = [(i*2) for i in range(1000000)]
    A = np.asarray(A)
    end = time.time()
    print ('time= ', end-start)
    
