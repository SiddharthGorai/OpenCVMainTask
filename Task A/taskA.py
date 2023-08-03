import cv2
import numpy as np

img = cv2.imread("Task.png",1)
# img = cv2.resize(img, (0,0),fx=0.8,fy=.8)
height, width, _ = img.shape
# print(height, width)

pt = (width - 10, height - 10)

# Horizontal Lines
c = int((height - 20)/5)
thickness = 2
for i in range(6):
    cv2.line(img, (pt[0]- i*c,pt[1]), (pt[0]- i*c,pt[1] - height + 20 ), 
    (0,255,0), thickness)


# Vertical Lines

for i in range(6):
    cv2.line(img, (pt[0],pt[1] - i*c), (pt[0] - width + 20,pt[1] - i*c), 
             (0,255,0), thickness)

# red solid lines
cv2.line(img, (pt[0] - 4*c ,pt[1] - 4*c), (pt[0] - 4*c,pt[1] - 3*c), 
             (0,0,255), thickness)

cv2.line(img, (pt[0] - 1*c ,pt[1] - 4*c), (pt[0] - 1*c,pt[1] - 3*c), 
             (0,0,255), thickness)

# red verticle dotted lines

cv2.line(img, (pt[0] - 4*c ,pt[1] - 4*c), (pt[0] - 4*c,pt[1] - 5*c), 
             (0,0,255), 2)
cv2.line(img, (pt[0] - 1*c ,pt[1] - 4*c), (pt[0] - 1*c,pt[1] - 5*c), 
             (0,0,255), 2)

# red horizontal dotted lines

cv2.line(img, (pt[0] - 4*c ,pt[1] - 4*c), (pt[0] - 3*c,pt[1] - 4*c), 
             (0,0,255), 2)
cv2.line(img, (pt[0] - 4*c ,pt[1] - 5*c), (pt[0] - 3*c,pt[1] - 5*c), 
             (0,0,255), 2)
cv2.line(img, (pt[0] - 4*c ,pt[1] - 2*c), (pt[0] - 3*c,pt[1] - 2*c), 
             (0,0,255), 2)
cv2.line(img, (pt[0] - 4*c ,pt[1] - 1*c), (pt[0] - 3*c,pt[1] - 1*c), 
             (0,0,255), 2)

cv2.imwrite("Result.png",img)
cv2.imshow("img",img)
cv2.waitKey(0)