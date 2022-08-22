import face_recognition as fr
import os
import cv2
impath = "Personas"
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0
for imagenes in os.listdir(impath):
    print(imagenes)
    image = cv2.imread(impath + "/" + imagenes)
    faces = faceClassif.detectMultiScale(image, 1.1, 5)
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, (150, 150))
        cv2.imwrite("faces/"+str(count)+".jpg", face)
        count += 1
        #cv2.imshow("rostro", face)
        # cv2.waitKey(0)
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
    #cv2.imshow("imagnes", image)
   # cv2.waitKey(0)
# cv2.destroyAllWindows()
