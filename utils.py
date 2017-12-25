import cv2
import numpy as np
from os import listdir
from os.path import isfile, splitext
from data_generator import DataGenerator

def checking_file_exist(link):
    if not isfile(link):
        print("File is not avalable: ", link)
        return False
    return True

#Create folder name
#Tạo tên folder từ thứ tự của folder
#Trong tập UCSDpred1 có 34 folder với tên gọi Train<số thứ tự>
#Ví dụ folder thứ 34 có tên Train034
def get_folder_name(i):
    fname = "Train"
    if i <10:
        fname = fname + "00" + str(i)
    elif i<100:
        fname = fname + "0" + str(i)
    return fname

#Create file name
#Tạo tên fole từ thứ tự của file
#Trong tập UCSDpred1 mỗi folder có 200 frame ảnh, mỗi ảnh có tên gọi <số thứ tự>.tif
#Ví dụ frame ảnh thứ 200 có tên 200.tif
def get_file_name(i):
    if i < 10:
        fname = "00" + str(i) + ".tif"
    elif i < 100:
        fname = "0" + str(i) + ".tif"
    else:
        fname = str(i) + ".tif"
    return fname

def is_image(file_name):
    name, ext = splitext(file_name)
    if ext==".jpg":
        return True
    return False


class LK_Optical_Flow:
    def __init__(self, width, height, winSize=20):
        self.width = width
        self.height = height
        self.winSize = winSize
        self.max_distance = np.sqrt(2)*self.winSize
        self.prevPts = [[(int(i)%self.width, int(i/self.width))] for i in range(self.width*self.height)]
        self.prevPts = np.asarray(self.prevPts, dtype='float32')

    def cast(self, x):
        x[np.where(x>255)] = 255
        return x

    def calc(self, frame1, frame2):
        H = self.height
        W = self.width

        nextPts, status, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, self.prevPts, None, winSize=(self.winSize, self.winSize))
        
        dx = nextPts[:,0,0]-self.prevPts[:,0,0]
        dy = nextPts[:,0,1]-self.prevPts[:,0,1]

        angle = np.arctan2(dy, dx)*180/np.pi
        angle[np.where(angle<0)] += 360

        distance = np.sqrt(dx*dx + dy*dy)
        
        Hue = (self.cast(angle*255/360)*status[:,0]).astype('uint8').reshape((H,W,1))
        Sar = (self.cast(distance*255/self.max_distance)*status[:,0]).astype('uint8').reshape((H,W,1))
        Val = np.full_like(Hue, 255)
        HSV = np.concatenate((Hue,Sar,Val), axis=2)
        return HSV

def get_list_sample(batch_folder):
    list_file = []
    for f in listdir(batch_folder):
        file_path = batch_folder + "/" + f
        if isfile(file_path) and is_image(file_path):
            list_file.append(file_path)
    return list_file

def get_dataset(folder):
    print ("Check files")
    list_file = get_list_sample(folder)
    print ("Loading dataset")
    X = np.empty((len(list_file), 32, 32, 3))
    Y = np.ones((len(list_file), 1))
    for i in range(len(list_file)):
        print(list_file[i])
        img = cv2.imread(list_file[i], cv2.IMREAD_ANYCOLOR)
        X[i] = img
    return X, Y


    

