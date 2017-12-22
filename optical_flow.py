import cv2
import numpy as np
import time

import LK_optical_flow as LK 

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    frame1 = None
    frame2 = None
    mLKOF = LK.LK_Optical_Flow(238,158)
    while(True):
        #Save last frame
        if frame2 is not None:
            frame1 = np.copy(frame2)

        #Capture current frame
        ret, frame2 = cap.read()
        print (frame2.shape)

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

    