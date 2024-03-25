#1-configure webcam with bycharm
import math
import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
import pandas as ps
license_plate_detector = YOLO('./models/car_Plate_reconizer.pt')

#cap = cv2.VideoCapture(0) #a function in open cv library for webcam ...0 for first cam.. 1 for second cam etc..
cap = cv2.VideoCapture("images/1.mp4")
#2- run webcam with yolo
model = YOLO("../Yolo-Weights/yolov8n.pt")
classNames = ['-', '0', '1', '2', '3', '3een', '4', '5', '6', '7', '8', '9', 'alf', 'beh', 'dal', 'fehh', 'gem', 'heh', 'lam', 'mem',
 'noon', 'noonnoon', 'qaf', 'reh', 'sad', 'seen', 'tah', 'waw', 'yeeh', 'yeh']
df = ps.DataFrame(columns=['Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])
while True:#to capture every single frame of vedio frames
    success, img = cap.read()
    results = license_plate_detector(img, stream=True)
    #create a border sourouding the objects
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0] #points of boundary box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))#borders
            #compute accurcy
            conf = math.ceil((box.conf[0] * 100)) / 100
            #print class name above object border
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale = 0.7, thickness = 1)
            df = df._append({'Class': classNames[cls], 'Confidence': conf, 'X1': x1, 'Y1': y1, 'X2': x2, 'Y2': y2},ignore_index=True)


    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xff == ord('q'): #to stop webcam click q
        break
#another way to stop webcam
cap.release()
cv2.destroyAllWindows()
df.to_excel('images/detections3.xlsx', index=False)