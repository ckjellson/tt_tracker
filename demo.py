import numpy as np
import analysis_functions

'''
Creates an "analyzer" object for visualization of processed video data

Specify:
    vidname: name of the instance to use
'''


vidname = 'outside'

c1 = np.load('data/'+vidname+'/c1.npy')
c2 = np.load('data/'+vidname+'/c2.npy')

ball_pos1 = np.load('data/' + vidname + '/ballpath1.npy')
param1 = np.load('data/' + vidname + '/param1.npy')
height1 = param1[0]
width1 = param1[1]
fps1 = param1[2]

ball_pos2 = np.load('data/' + vidname + '/ballpath2.npy')
param2 = np.load('data/' + vidname + '/param2.npy')
height2 = param2[0]
width2 = param2[1]
fps2 = param2[2]

analyzer = analysis_functions.analyzer(height1, width1, height2, width2, c1, c2, np.transpose(ball_pos1), np.transpose(ball_pos2), fps1)

# Available functions to run:
analyzer.visualize_3d_strokes(1)      # Plots one stroke at a time in 3d in point nbr 1
# analyzer.visualize_2d_strokes(1)      # Plots one stroke at a time in 2d in point nbr 1
# analyzer.animate_3d_path()            # Animates the 3d path taken in the whole video
# analyzer.bounce_heatmap()             # Plots a heatmap of the detected bounces
# analyzer.plot_3d_point(1)             # Plots the trajectory of point 1
# analyzer.plot_trajectory(2)           # Plots all trajectories from camera 1/2 perspective