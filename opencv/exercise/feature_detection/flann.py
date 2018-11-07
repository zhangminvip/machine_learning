import numpy as np
import cv2
from matplotlib import pyplot as plt

queryImage = cv2.imread('/home/minzhang/ml/image/defu.png', cv2.IMREAD_GRAYSCALE)
trainImage = cv2.imread('/home/minzhang/ml/image/defu2.jpg', cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(queryImage, None)
kp2, des2 = sift.detectAndCompute(trainImage, None)

FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)

flann = cv2.FlannBasedMatcher(indexParams, searchParams)
matches = flann.knnMatch(des1, des2, k=2)
matchesMask = [[0, 0] for i in range(len(matches))]

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]

drawParams = dict(matchColor=(0, 255, 0),
                  singlePointColor=(255, 0, 0),
                  matchesMask=matchesMask,
                  flags=0
                  )

resultImage = cv2.drawMatchesKnn(queryImage, kp1, trainImage, kp2, matches, None, **drawParams)
plt.imshow(resultImage)
plt.show()
