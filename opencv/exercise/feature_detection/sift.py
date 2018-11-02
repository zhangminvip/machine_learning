import cv2
import sys
import numpy as np

# print(cv2.getBuildInformation())
# print(cv2.__version__,cv2.__file__)
# imgpath = '/home/minzhang/ml/Varese.jpg'
imgpath = sys.argv[1]
alg = sys.argv[2]

def fg(algorithm):
    if algorithm == 'SIFT':
        return cv2.xfeatures2d.SIFT_create()
    if algorithm == 'SURF':
        return cv2.xfeatures2d.SURF_create(float(sys.argv[3]) if len(sys.argv)== 4 else 4000)
img = cv2.imread(imgpath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fd_flag = fg(alg)
keypoints, descriptor = fd_flag.detectAndCompute(gray,None)
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
                        color=(51,163,236))
cv2.namedWindow('sift',cv2.WINDOW_NORMAL)
cv2.resizeWindow('sift', 700, 700)
cv2.imshow('sift', img)

while(True):
    if cv2.waitKey(int(1000 / 12)) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
print('wo')