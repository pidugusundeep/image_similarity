import numpy as np
import cv2

cap = cv2.VideoCapture('/home/andrei/temp/video/Smesne macke - funny cats.webm')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        
        cv2.imshow('frame', frame)
        # & 0xFF is required for a 64-bit system
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
