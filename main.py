#Face Recognizing and Face-Hand Landmarks

import cv2, numpy, sys, os
import mediapipe as mp

#Initilazing Objects
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

drawing_spec = mp_drawing.DrawingSpec(thickness= 1, circle_radius= 1)
webcam = cv2.VideoCapture(0)

#defining for face recognize
size = 4
haarFile = "haarcascade_frontalface_default.xml"
datasets = "datasets"

print("Recognizing Face Please Be in sufficient Light")

#create a list of images and a list of corresponding names
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):

    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)

        for filename in os.listdir(subjectpath):
            path = subjectpath + "/" + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))


        id += 1

(width , height)=(130,100)

#Create a Numpy array from the two lists above
(images , labels) = [numpy.array(lis) for lis in [images , labels]]

#OpenCV trains a model from the images
#NOTE FOR OpenCV': remove '.face'
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haarFile)


with mp_face_mesh.FaceMesh(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as face_mesh:
    while True:
        success , image = webcam.read()
        gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3 , 5)

        #convert the color space from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #To improve performance
        image.flags.writeable = False

        #Detect the face and hand landmarks
        results = face_mesh.process(image)
        resultsHand = holistic_model.process(image)

        #To improve performance
        image.flags.writeable = True

        #convert backk to the BGR color
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #Draw the face mesh annotations on the image
        if results.multi_face_landmarks:
            for face_landmmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image = image,
                    landmark_list= face_landmmarks,
                    connections= mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec= None,
                    connection_drawing_spec= mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )
                # hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    resultsHand.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )

                mp_drawing.draw_landmarks(
                    image,
                    resultsHand.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )




        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face,(width, height))

            #Try to recognize the face
            prediction = model.predict(face_resize)
            cv2.rectangle(image, (x,y),(x+w, y+h), 3)

            if prediction[1] / 500 < 1 and prediction[1] < 85:
                cv2.putText(image, "% s - %.0f" %
                            (names[prediction[0]], prediction[1]), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0))

            elif prediction[1] < 500 and prediction[1] >= 85:
                cv2.putText(image, "not recognized", (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

            else:
                cv2.putText(image, "not recognized",
                            (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)

                            )
        #display
        cv2.imshow("FaceDetect", image)

        if cv2.waitKey(10) == 27:
            break

