import cv2
import numpy as np

# Everything commented away has to do with breakpoints, shouldn't have to do this!

path1 = "videos_original\out_a_full.mp4"
path2 = "videos_original\out_b_full.mp4"
vidname = 'outside'

cap1 = cv2.VideoCapture(path1)
nbr_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))-1
cap2 = cv2.VideoCapture(path2)
nbr_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))-1

_, f1 = cap1.read()
_, f2 = cap2.read()
height,width,channels = f1.shape

# Find a starting point
while True:
    f = np.hstack((f1, f2))
    f = cv2.resize(f, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('',f)
    k = cv2.waitKey(0) & 0xFF
    if k==49: # 1 is pressed
        _, f1 = cap1.read()
        nbr_frames1 -= 1
    elif k==50: # 2 is pressed
        _, f2 = cap2.read()
        nbr_frames2 -= 1
    else:
        break

# Create and save two equally long clips
clip1 = cv2.VideoWriter('videos/' + vidname + '1.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width,height))
clip2 = cv2.VideoWriter('videos/' + vidname + '2.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width,height))

accum = 1
capture = True
for i in range(min([nbr_frames1, nbr_frames2])):
    _, f1 = cap1.read()
    _, f2 = cap2.read()
    clip1.write(f1)
    clip2.write(f2)
clip1.release()
clip2.release()
cap1.release()
cap2.release()
cv2.destroyAllWindows()
