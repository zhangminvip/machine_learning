import cv2


def generate():
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)
    count = 0
    while (True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            f = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            cv2.imwrite('/home/minzhang/ml/code/opencv/face_recognition/data/at/jm/%s.pgm' % str(count), f)
            count += 1

        cv2.imshow('camera', frame)
        if cv2.waitKey(int(1000 / 12)) & 0xff == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


generate()
