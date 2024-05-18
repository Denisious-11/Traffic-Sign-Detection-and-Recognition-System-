from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
import cv2 as cv
import numpy as np
import cv2
import math
import pyttsx3 
import os
from classes import *
from matplotlib import pyplot as plt
import numpy as np


a = Tk()
a.title("Traffic Sign Detector")
a.geometry("750x600")
a.maxsize(750,600)
a.minsize(750,600)


# Load Yolo
print("LOADING YOLO")
net = cv2.dnn.readNet("Trained_Models/yolov3_r_60000.weights", "Trained_Models/yolov3_r.cfg")
classes=get_classes()

# init function to get an engine instance for the speech synthesis
engine = pyttsx3.init()

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
            cv2.putText(yimg, label, (x, y-15), font, 1, color, 2)\
            # say method on the engine that passing input text to be spoken
            engine.say(label)
            # run and wait method, it processes the voice commands.
            engine.runAndWait()


    cv2.imshow("Testing", yimg)
    cv2.waitKey(1)
   


def start():
    cap = cv2.VideoCapture(0) 
    while(cap.isOpened()):
        ret, frame = cap.read()
        detect_tsignals(frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    
    cap.release()
    cv2.destroyAllWindows()


   
def Check():
    global f
    f.pack_forget()

    f = Frame(a, bg="white")
    f.pack(side="top", fill="both", expand=True)

    global f1
    f1 = Frame(f, bg="light goldenrod")
    f1.place(x=0, y=0, width=760, height=620)
    f1.config()

    

    obj_button = Button(
        f1, text="Test Here", width=20,height=3, command=lambda: start(), bg="hot pink")
    obj_button.place(x=300,y=200)



def Home():
    global f
    f.pack_forget()

    f = Frame(a, bg="salmon")
    f.pack(side="top", fill="both", expand=True)

    front_image = Image.open("Screenshots/home.jpg")
    front_photo = ImageTk.PhotoImage(front_image.resize((a.winfo_width(), a.winfo_height()), Image.ANTIALIAS))
    front_label = Label(f, image=front_photo)
    front_label.image = front_photo
    front_label.pack()

    home_label = Label(f, text="Traffic Sign Detector",
                       font="arial 35", bg="white")
    home_label.place(x=190, y=110)


f = Frame(a, bg="salmon")
f.pack(side="top", fill="both", expand=True)
front_image1 = Image.open("Screenshots/home.jpg")
front_photo1 = ImageTk.PhotoImage(front_image1.resize((750, 600), Image.ANTIALIAS))
front_label1 = Label(f, image=front_photo1)
front_label1.image = front_photo1
front_label1.pack()

home_label = Label(f, text="Traffic Sign Detector",
                   font="arial 35", bg="white")
home_label.place(x=190, y=110)

m = Menu(a)
m.add_command(label="Home", command=Home)
checkmenu = Menu(m)
m.add_command(label="Test", command=Check)
a.config(menu=m)


a.mainloop()
