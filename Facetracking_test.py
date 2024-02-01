# python3.10 -m venv studysession
import URBasic
import math
import numpy as np
import sys
import cv2
import time
import imutils
from imutils.video import VideoStream
import math3d as m3d
import mediapipe as mp
import CAMTracking.FaceTracking1 as FT

import matplotlib
matplotlib.use('MacOSX')
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

print(sys.platform)
frameWidth = 1920
frameHeight = 1080
cap = cv2.VideoCapture(1)
cap.set(1920, frameWidth)
cap.set(1080, frameHeight)
#cap.set(100,150)

face_tracker = FT.FaceTracking()
features_to_track = ['lips', 'nose', 'left_eye', 'right_eye']
'''
class plotfigure:

    def __init__(self):

        self.fig, self.ax = plt.subplot()
        self.ax.set_aspect('equal')
        #self.ax.set_xlim(0, 1000)
        #self.ax.set_ylim(0, 1000)
        self.ax.hold(True)
        self.b1_points
        self.b2_points
        self.b3_points
        self.b4_points
        self.background

    def plot_setup(self):
        plt.show()
        plt.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def data_plot(self,b1,b2,b3,b4):
        self.b1_points = self.ax.plot(b1[0], b1[1], 'o')[0]
        self.b2_points = self.ax.plot(b2[0], b2[1], 'o')[0]
        self.b3_points = self.ax.plot(b3[0], b3[1], 'o')[0]
        self.b4_points = self.ax.plot(b4[0], b4[1], 'o')[0]

    def data_plot_update(self,b1,b2,b3,b4):
         self.b1_points.set_data(b1[0], b1[1])
         self.b2_points.set_data(b2[0], b2[1])
         self.b3_points.set_data(b3[0], b3[1])
         self.b4_points.set_data(b4[0], b4[1])
         self.fig.canvas.restore_region(self.background)
         self.ax.draw_artist(self.b1_points)
         self.ax.draw_artist(self.b2_points)
         self.ax.draw_artist(self.b3_points)
         self.ax.draw_artist(self.b4_points)
'''

def init():
    ax.set_xlim(-1000.1000)
    ax.set_ylim(-1000, 1000)
    return ln,

def update(b1,b2,b3,b4):
    #xdata.append(frame)
    #ydata.append(np.sin(frame))
    #ln.set_data(xdata, ydata)
    ln.set_data(b1[0], b1[1])
    ln.set_data(b2[0], b2[1])
    ln.set_data(b3[0], b3[1])
    ln.set_data(b4[0], b4[1])
    print("b1[0]=",b1[0],"b1[1]=",b1[1])
    return ln,

#ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
#                    init_func=init, blit=True)
#plt.show()


#plt_figure = plotfigure()
#plt_figure.plot_setup()

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')
while True:

    result, video_frame = cap.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully


    # Create an instance of the FaceTracking class
    #faces = face_detect_fun(video_frame)  # apply the function we created to the video frame
    #faces,center_point = face_tracker.face_center_detection(video_frame)
    #faces,center_point = face_tracker.track_specific_features(video_frame,features_to_track)
    faces, center_point, b1, b2, b3, b4 = face_tracker.face_orientation(video_frame)
    ani = FuncAnimation(fig, update(b1,b2,b3,b4), init_func=init, blit=True)
    #plt.show()
    #faces = face_tracker.face_landmark(video_frame)
    #print(center_point)

    #print(center_point)

    cv2.imshow(
        "My Face Detection Project", faces
    )  # display the processed frame in a window named "My Face Detection Project"

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()