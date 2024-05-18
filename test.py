import cv2
import numpy as np
from classes import *

# Load Yolo
print("LOADING YOLO")
net = cv2.dnn.readNet("Trained_Models/yolov3_r_60000.weights", "Trained_Models/yolov3_r.cfg")
classes=get_classes()

layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def detect_tsignals(yimg):
    # print(yimg)
    yimg = cv2.resize(yimg, None, fx=0.7, fy=0.7)
    height, width, channels = yimg.shape
    # print(height,width,channels)

    # Detecting objects
    blob = cv2.dnn.blobFromImage(yimg, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # print(confidence)
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    labels=[]
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            print(label)
            labels.append(label)
            color = colors[i]
            cv2.rectangle(yimg, (x-10, y-10), (x + w+20, y + h+20), color, 2)
            cv2.putText(yimg, label, (x, y-15), font, 1, color, 2)


    cv2.imshow("Image", yimg)
    cv2.waitKey(1)
    return labels


def video1():

    cap = cv2.VideoCapture(0) 
    while(cap.isOpened()):
        ret, frame = cap.read()
        detect_tsignals(frame)

    
    cap.release()
    cv2.destroyAllWindows()

video1()