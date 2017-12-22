import cv2
import numpy as np
import time

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

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    frame1 = None
    frame2 = None
    mLKOF = LK_Optical_Flow(238, 158)
    while(True):
        #Save last frame
        if frame2 is not None:
            frame1 = np.copy(frame2)

        #Capture current frame
        ret, frame2 = cap.read()

        #Conver to Gray
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.resize(frame2, (238, 158))
        

        #Calculating optical flow between frame1 and frame2
        if frame1 is not None:
            start = time.time()
            hsv = mLKOF.calc(frame1, frame2)
            end = time.time()
            print('time= ', end-start)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Display the resulting frame
            cv2.imshow('frame', bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
