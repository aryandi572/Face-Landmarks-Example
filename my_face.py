import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    print('get frame',frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        print('value x1',x1)
        y1 = face.top()
        print('value y1',y1)
        x2 = face.right()
        print('value x2',x2)
        y2 = face.bottom()
        print('value y2',y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)
        print('landmarks',landmarks)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    
cv2.destroyAllWindows()
cap.release()