import cv2

day = cv2.imread('../data/train/00000850/Day/20151101_142506.jpg')
day_g = cv2.cvtColor(day, cv2.COLOR_BGR2GRAY)
cv2.imwrite("testImgs/day.png", day_g)