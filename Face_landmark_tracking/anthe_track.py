#import required libraries
import numpy as np
import cv2
import datetime
import URBasic
import math
import sys
import time
import imutils
from imutils.video import VideoStream
import math3d as m3d
import mediapipe as mp
import CAMTracking.FaceTracking1 as FT

#Creates a video capture object 
#to play the video from a file
video = cv2. VideoCapture("Anthe.mp4")
#check if the video file is opened successfully
face_tracker = FT.FaceTracking()
features_to_track = ['lips', 'nose', 'left_eye', 'right_eye']
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
imgH=480
imgW=640
video_write=cv2.VideoWriter('video_opencv.avi',fourcc,20,(imgW, imgH))
while(video.isOpened()):
    # read the frame from the video file
    ret, video_frame=video.read()
    # if the frame was captured successfully
        # print the current datetime
        # display the frame
        
    faces,center_point = face_tracker.track_specific_features(video_frame,features_to_track)
    cv2.imshow("Source", faces)
    video_write.write(faces)
    key = cv2.waitKey(30)
    # if key q is pressed then break 
    if key == 113:
        break 
        
#Closes video file or capturing device.
video.release()
#finally destroy/close all open windows
cv2.destroyAllWindows()