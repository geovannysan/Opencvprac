from pickle import TRUE
import face_recognition
import os
import cv2
imagenpaht = "faces"
facesEcoding = []
faceNAme = []

for filename in os.listdir(imagenpaht):
    imagen = cv2.imread(imagenpaht + "/" + filename)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    if len(imagen) == 0:
        break
    face_code = face_recognition.face_encodings(
        imagen, known_face_locations=[(0, 150, 150, 0)])[0]
    facesEcoding.append(face_code)
    faceNAme.append(filename.split(".")[0])

    # print(facesEcoding)

    #cv2.imshow("caras", image)
    # cv2.waitKey(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    frame = cv2.flip(frame, 1)
    orig = frame.copy()
    faces = faceClassif.detectMultiScale(frame, 1.25, 5)
    for (x, y, w, h) in faces:
        face = orig[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        actual_rosto = face_recognition.face_encodings(
            face, known_face_locations=[(0, w, h, 0)])[0]
        resulta = face_recognition.compare_faces(facesEcoding, actual_rosto)

        if True in resulta:
            index = resulta.index(True)
            name = faceNAme[index]
            color = (125, 220, 0)
        else:
            name = "Descoocido"
            color = (50, 50, 255)
        cv2.rectangle(frame, (x, y+h), (x+w, y+h+30), color, -1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y + h + 25), 2, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Video", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cap.release()
cap.destroyAllWindows()
