import numpy as np
import cv2
import analysis_functions
import process_video as pv
import time
import os

##
vidname = 'bengt'
newvid = True          # If true creates new data folder
select_c = True        # True=select new corners, double click on points from top left clockwise, then net bottom to top
calc_trace = True       # True=find ball trace
flipped1 = False        # Change if the video is upside down
flipped2 = False

if newvid:
    os.mkdir('data/'+vidname)
vid1_path = 'videos/'+vidname+'1.mp4'
vid2_path = 'videos/'+vidname+'2.mp4'
if select_c:
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
else:
    c1 = np.load('data/'+vidname+'/c1.npy')
    c2 = np.load('data/'+vidname+'/c2.npy')
##
if calc_trace:
    t1 = time.time()
    height1, width1, ball_pos1, fps1 = pv.read_video(vid1_path,flipped1)
    t2 = time.time()
    param1 = np.array([height1,width1,fps1])
    np.save('data/'+vidname+'/ballpath1',ball_pos1)
    np.save('data/'+vidname+'/param1',param1)
    print('Parameters:')
    print(param1)
    print('Time:{}'.format(t2-t1))
else:
    ball_pos1 = np.load('data/'+vidname+'/ballpath1.npy')
    param1 = np.load('data/'+vidname+'/param1.npy')
    height1 = param1[0]
    width1 = param1[1]
    fps1 = param1[2]
##
if calc_trace:
    t1 = time.time()
    height2, width2, ball_pos2, fps2 = pv.read_video(vid2_path,flipped2)
    t2 = time.time()
    param2 = np.array([height2,width2,fps2])
    np.save('data/'+vidname+'/ballpath2',ball_pos2)
    np.save('data/'+vidname+'/param2',param2)
    print('Parameters:')
    print(param2)
    print('Time:{}'.format(t2-t1))
else:
    ball_pos2 = np.load('data/'+vidname+'/ballpath2.npy')
    param2 = np.load('data/'+vidname+'/param2.npy')
    height2 = param2[0]
    width2 = param2[1]
    fps2 = param2[2]
##
analyzer = analysis_functions.analyzer(height1, width1, height2, width2, c1, c2, np.transpose(ball_pos1), np.transpose(ball_pos2), fps1)#,breakpoints)
##
# analyzer.plot_trajectory(1)
# analyzer.plot_3d_point(1)
# analyzer.visualize_3d_strokes(1)
# analyzer.visualize_2d_strokes(1)
# analyzer.plot_bounce_points()
# analyzer.animate_3d_path()
analyzer.bounce_heatmap()

##

