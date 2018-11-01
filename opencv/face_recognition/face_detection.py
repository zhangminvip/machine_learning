import cv2
import os
import numpy as np
import sys


def detect():
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)
    while (True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (ex + x, ey + y), (ex + x + ew, ey + y + eh), (0, 255, 0), 2)
        cv2.imshow('camera', frame)
        if cv2.waitKey(int(1000 / 12)) & 0xff == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()


def read_images(path, names,sz=None):
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    print(filename,subdirname)
                    if filename == '.directory':
                        continue
                    filepath = os.path.join(subject_path, filename)
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    if sz is not None:
                        im = cv2.resize(im, (200, 200))
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                    # names[c] = subdirname
                except IOError:
                    print('IO error({0}{1})'.format(IOError.errno, IOError.strerror))
                except:
                    print('unexcept error', sys.exc_info()[0])
                    raise
            # print(c)
            names[c] = subdirname
            c += 1
    print(names)
    return [X, y],names


def face_rec():
    names = {}
    if len(sys.argv) < 2:
        print('USAGE: facerec_demo.py <path to images> [path to store images at]')
        sys.exit()
    [X, y],names = read_images(sys.argv[1],names)
    # print(X)
    print(y)
    y = np.asarray(y, dtype=np.int32)
    if len(sys.argv) == 3:
        outdir = sys.argv[2]
    model = cv2.face.EigenFaceRecognizer_create()
    # model = cv2.face.FisherFaceRecognizer_create()
    model.train(np.asarray(X), np.asarray(y))
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    while (True):
        read, img = camera.read()
        img = np.fliplr(img).copy()
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[x:x + w, y:y + h]
            try:
                # print('...')
                roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                params = model.predict(roi)
                print(params[0])
                cv2.putText(img, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            except:
                continue
        cv2.imshow('camera', img)
        # print(params[0])
        if cv2.waitKey(int(1000 / 12)) & 0xff == ord('q'):
            break
        # print('1')
    camera.release()
    cv2.destroyAllWindows()


# detect()


face_rec()
