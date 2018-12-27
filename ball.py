# -*- coding: utf-8 -*-
"""
Author: Jason Chow
Ball and cup tracking algorithm using OpenCV used in "Party Pong" scoring system
"""

#import stuff
from collections import deque
import cv2
import imutils
import numpy as np

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
            
    cv2.drawContours(frame, contour_list, -1, (255,100,25),2)
    return frame        


pinklower = (15, 230, 25)
pinkupper = (25, 255,255)
pts = deque(maxlen=25)


video_capture = cv2.VideoCapture(0)


while True:
    
    _, frame = video_capture.read()
    
    blurred =   cv2.GaussianBlur(frame, (11,11), 0)
    hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
         
    mask = cv2.inRange(hsv, pinklower, pinkupper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    center = None
    
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
		# only proceed if the radius meets a minimum size
        if radius > 5:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points	
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            
    pts.appendleft(center)
    
    for i in range (1,len(pts)):
        if pts[i-1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt((len(pts))/float(i+1))*2.5) 
        cv2.line(frame,pts[i-1],pts[i], (255,0,0), thickness)
    
    frame = shape_detect(frame)
    
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
video_capture.release()
cv2.destroyAllWindows()


