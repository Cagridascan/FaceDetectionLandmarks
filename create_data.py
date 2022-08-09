# Creating data from webcam


#importing the libraries
import cv2, sys, numpy, os

#address of the haar file
haarFile = "haarcascade_frontalface_default.xml"


#datasets file directory
datasets = "datasets"

#for data directory name
sub_data = input("Enter the name of the data:")

#create and check the file path
path = os.path.join(datasets, sub_data)

if not os.path.isdir(path):
    os.mkdir(path)

#defining the weight and height
(width, height) = (130, 100)


face_cascade = cv2.CascadeClassifier(haarFile)
webcam = cv2.VideoCapture(0) #'0' for webcam

#loop until it has 30 images
count = 1
while count<30:
    #taking the image from webcam
    (_, image) = webcam.read()

    #convert the color space from BGR to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:

        cv2.rectangle(image,(x,y),(x+w, y+h),(255,0,0),2)
        face = gray[y:y + h, x:x+w]

        face_resize = cv2.resize(face, (width,height))
        cv2.imwrite("% s/% s.png" % (path,count),face_resize)

    count += 1

    cv2.imshow("CreateData",image)

    key = cv2.waitKey(10)
    if key == 27:
        break