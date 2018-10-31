import cv2

filename = '/home/minzhang/ml/hpf_test.jpg'


def detect(filename):
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
    # cv2.namedWindow('beauty detected')
    cv2.imshow('face detected',img)
    cv2.imwrite('./face_detected.jpg',img)
    cv2.waitKey()


detect(filename)
