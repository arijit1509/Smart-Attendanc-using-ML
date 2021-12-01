import cv2
import numpy as np
import face_recognition

imgArjun = face_recognition.load_image_file('ImageBasic/Arjun Munda.jpg')
imgArjun = cv2.cvtColor(imgArjun,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImageBasic/Arjun Test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgArjun)[0]
encodeArjun = face_recognition.face_encodings(imgArjun)[0]
cv2.rectangle(imgArjun,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

facelocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeArjun],encodeTest)
faceDis = face_recognition.face_distance([encodeArjun],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Arjun Munda',imgArjun)
cv2.imshow('Arjun Test',imgTest)
cv2.waitKey(0)