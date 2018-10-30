import cv2
import numpy as np

img = cv2.imread('/home/minzhang/ml/hpf_test.jpg')
cv2.imwrite('test.jpg',cv2.Canny(img,200,300))
cv2.imshow("canny", cv2.imread('test.jpg'))
cv2.waitKey()
cv2.destroyAllWindows()