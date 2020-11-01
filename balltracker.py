import numpy as np
import cv2
import process_video as pv
import time
import os

'''
Loads two synced videos and generates an interface for choosing corner and net
positions, and then processes the videos, i.e. detects the position of a table
tennis ball in each frame.

Specify:
    vidname: name of the instance to be processed
'''

vidname = 'outside'
flipped1 = False        # Change if the video is upside down
flipped2 = False

os.mkdir('data/'+vidname)
vid1_path = 'videos/'+vidname+'1.mp4'
vid2_path = 'videos/'+vidname+'2.mp4'

# Select corners and net position
cap1 = cv2.VideoCapture(vid1_path)
ret1, frame = cap1.read()
h1,w1,c1 = frame.shape

class CoordinateStore:
    def __init__(self):
        self.points = []

    def select_point(self,event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                cv2.circle(frame,(x,y),3,(255,0,0),-1)
                self.points.append((x,y))

#instantiate class
coordinateStore1 = CoordinateStore()
# Create a black image, a window and bind the function to window
cv2.namedWindow('image')
cv2.setMouseCallback('image',coordinateStore1.select_point)
while True:
    cv2.imshow('image',frame)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
print("Selected Coordinates: ")
c1 = [[p[0],h1-p[1],1] for p in coordinateStore1.points]
c1 = np.array(c1)
print(c1)
cap1.release()

cap2 = cv2.VideoCapture(vid2_path)
ret2, frame = cap2.read()
h2,w2,c2 = frame.shape

#instantiate class
coordinateStore1 = CoordinateStore()
# Create a black image, a window and bind the function to window
cv2.namedWindow('image')
cv2.setMouseCallback('image',coordinateStore1.select_point)
while True:
    cv2.imshow('image',frame)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
print("Selected Coordinates: ")
c2 = [[p[0],h2-p[1],1] for p in coordinateStore1.points]
c2 = np.array(c2)
print(c2)
cap2.release()

np.save('data/'+vidname+'/c1',c1)
np.save('data/'+vidname+'/c2',c2)

# Process video 1
t1 = time.time()
height1, width1, ball_pos1, fps1 = pv.read_video(vid1_path,flipped1)
t2 = time.time()
param1 = np.array([height1,width1,fps1])
np.save('data/'+vidname+'/ballpath1',ball_pos1)
np.save('data/'+vidname+'/param1',param1)
print('Parameters:')
print(param1)
print('Time:{}'.format(t2-t1))

# Process video 2
t1 = time.time()
height2, width2, ball_pos2, fps2 = pv.read_video(vid2_path,flipped2)
t2 = time.time()
param2 = np.array([height2,width2,fps2])
np.save('data/'+vidname+'/ballpath2',ball_pos2)
np.save('data/'+vidname+'/param2',param2)
print('Parameters:')
print(param2)
print('Time:{}'.format(t2-t1))

