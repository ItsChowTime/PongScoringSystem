# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 18:45:02 2018
'Image = cv2.imread(''Image.jpg')
    'Resize_Image = cv2.resize(Image, (684, 912))

@author: Jason
"""

import cv2


def shape_detect(frame):

    bilateral_filter = cv2.bilateralFilter(frame, 5, 175, 200)
    edge_detect = cv2.Canny(bilateral_filter, 75, 200)

    _, contours, _= cv2.findContours(edge_detect, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.025*cv2.arcLength(contour,True), True)
            area = cv2.contourArea(contour)
            if ((len(approx)> 7) & (area>50)):
                contour_list.append(contour)
            
    cv2.drawContours(frame, contour_list, -1, (255,0,0),2)
    return frame


video_capture = cv2.VideoCapture(1)
_, frame = video_capture.read()

while True:
    _, frame = video_capture.read()
    
    canvas = shape_detect(frame)
    cv2.imshow('yeet', canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()