import cv2
import numpy as np
from os import mkdir
from os.path import isfile, exists
import sys
import LK_Optical_Flow as LK

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

def cal_optical_flow(ucsd_pred1_link, output_link):
    #Checking if output_link is not exist, create ouput folder
    #Kiểm tra folder output có tồn tại không, nếu không thì tạo folder output
    if not exists(output_link):
        mkdir(output_link)

    #Tính optical flow trên ảnh 238x158
    mLKOF = LK.LK_Optical_Flow(238, 158)

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

if __name__ == '__main__':
    #cal_optical_flow("../UCSDped1/Train", "../UCSDped1/Train_")
    create_patch("../UCSDped1/Train_", "../UCSDped1/TrainBatch")

    print("Done")






