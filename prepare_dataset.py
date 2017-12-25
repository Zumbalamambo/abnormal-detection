import cv2
import numpy as np
from os import mkdir, listdir, remove, rename
from os.path import isfile, exists, splitext
import sys
from utils import *

def cal_optical_flow(ucsd_pred1_link, output_link):
    #Checking if output_link is not exist, create ouput folder
    #Kiểm tra folder output có tồn tại không, nếu không thì tạo folder output
    if not exists(output_link):
        mkdir(output_link)

    #Tính optical flow trên ảnh 238x158
    mLKOF = LK_Optical_Flow(238, 158)

    #Go through 34 folder of UCSDpred1
    #Duyệt qua 34 folder của tập UCSDpred1
    for i in range(1, 35):
        folder_name = get_folder_name(i)
        folder_out = output_link + "/" + folder_name
        if not exists(folder_out):
            mkdir(folder_out)
        
        #Go through 199 image in each folder
        #Duyệt qua 199 ảnh trong mỗi folder
        for j in range(1,200):
            fname1 = get_file_name(j)
            print(fname1)
            fname2 = get_file_name(j+1)

            fout = folder_out + "/" + fname1

            fpath1 = ucsd_pred1_link + "/" + folder_name + "/" + fname1
            fpath2 = ucsd_pred1_link + "/" + folder_name + "/" + fname2

            if checking_file_exist(fpath1) and checking_file_exist(fpath2):
                #Tính optical flow giữa ảnh thứ j và ảnh j+1, lưu vào thư mục output
                f1 = cv2.imread(fpath1, cv2.IMREAD_GRAYSCALE)
                f2 = cv2.imread(fpath2, cv2.IMREAD_GRAYSCALE)

                hsv = mLKOF.calc(f1, f2)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imwrite(fout, bgr)


def create_patch(uscd_pred1_optical_flow, output_link):
    #Checking if output_link is not exist, create ouput folder
    #Kiểm tra folder output có tồn tại không, nếu không thì tạo folder output
    if not exists(output_link):
        mkdir(output_link)

    #Go through 34 folder of UCSDpred1
    #Duyệt qua 34 folder của tập UCSDpred1
    for i in range(1, 35):
        #Go through 196 image in each folder
        #Duyệt qua 196 ảnh trong mỗi folder
        for j in range(1, 197):
            print("Folder :%s, file: %s" %(i, j))
            folder_name = get_folder_name(i)
            folder_out = output_link + "/" + folder_name
            if not exists(folder_out):
                mkdir(folder_out)

            fname1 = get_file_name(j)
            fname2 = get_file_name(j+1)
            fname3 = get_file_name(j+2)
            fname4 = get_file_name(j+3)

            fpath1 = uscd_pred1_optical_flow + "/" + folder_name + "/" + fname1
            fpath2 = uscd_pred1_optical_flow + "/" + folder_name + "/" + fname2
            fpath3 = uscd_pred1_optical_flow + "/" + folder_name + "/" + fname3
            fpath4 = uscd_pred1_optical_flow + "/" + folder_name + "/" + fname4

            if checking_file_exist(fpath1) and checking_file_exist(fpath2) and checking_file_exist(fpath3) and checking_file_exist(fpath4):
                f1 = cv2.imread(fpath1, cv2.IMREAD_ANYCOLOR)
                f2 = cv2.imread(fpath2, cv2.IMREAD_ANYCOLOR)
                f3 = cv2.imread(fpath3, cv2.IMREAD_ANYCOLOR)
                f4 = cv2.imread(fpath4, cv2.IMREAD_ANYCOLOR)
                print(f1.shape)
                #Create patch from 4 frames
                for m in range(0, int(238/16)):
                    for n in range(0, int(158/16)):
                        fout = folder_out + "/" + str(j) + "_" + str(m) + "_" + str(n) + ".jpg"
                        bgr = np.zeros((32,32,3))
                        x = m*16
                        y = n*16
                        bgr[0:16, 0:16, :] = f1[y:y+16, x:x+16, :]
                        bgr[0:16, 16:36, :] = f2[y:y+16, x:x+16, :]
                        bgr[16:32, 0:16, :] = f3[y:y+16, x:x+16, :]
                        bgr[16:32, 16:32, :] = f4[y:y+16, x:x+16, :]

                        cv2.imwrite(fout, bgr)

def del_white(batch_path):
    is_first = True
    for i in range(1, 35):
        folder_name = batch_path + "/" + get_folder_name(i) + "/"
        print('Folder: ', folder_name)
        for f in listdir(folder_name):
            fname = folder_name + f
            checking_file_exist(fname)
            if isfile(fname) and is_image(f):
                img = cv2.imread(fname, cv2.IMREAD_ANYCOLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                if not np.any(img[:,:,0:2]):
                    if is_first == True:
                        is_first = False
                    else:
                        remove(fname)
                        print("Removed: ", fname)

def rename_and_move(batch_path, folder_out):
    list_file = []
    #Get files path
    print("Loading data")
    for i in range(1, 35):
        print("Loaded folder: ", i)
        folder_name = get_folder_name(i)
        folder_path = batch_path + "/" + folder_name + "/"
        for f in listdir(folder_path):
            file_path = folder_path + f
            if checking_file_exist(file_path) and is_image(file_path):
                list_file.append(file_path)
    #Rename and move to the same folder
    if not exists(folder_out):
                mkdir(folder_out)
    for i in range(len(list_file)):
        out_path = folder_out + "/" + str(i+1) + ".jpg"
        rename(list_file[i], out_path)
        print("Moved: ", list_file[i])
         
    

if __name__ == '__main__':
    #cal_optical_flow("../UCSDped1/Train", "../UCSDped1/Train_")
    #create_patch("../UCSDped1/Train_", "../UCSDped1/TrainBatch")
    #del_white('../UCSDped1/TrainBatch')
    rename_and_move('../UCSDped1/TrainBatch', '../UCSDped1/TrainBatch')

    print("Done")






