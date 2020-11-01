import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.linalg import qr,svd
from scipy.signal import argrelextrema
import matplotlib.animation as animation
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter

xtable = 2.74
ytable = 1.525

class analyzer:

    '''
    Analyzer class for applying triangulation, finding 3D tracks, and visualization

    Args:
        height1 (int): Height dimension of camera 1
        height2 (int): Height dimension of camera 2
        width1 (int): Width dimension of camera 1
        width2 (int): Width dimension of camera 2
        corners1 (np.array): Positions of corners and net in
                                camera 1
        corners2 (np.array): Positions of corners and net in
                                camera 2
        ball_pos_1 (np.array): Detected positions of ball in
                                camera 1
        ball_pos_2 (np.array): Detected positions of ball in
                                camera 2
        fps (int): Frames per second used in both cameras

    Attributes:
        fps     Frames per second
        h1      Height camera 1
        h2      Height camera 2
        w1      Width camera 1
        w2      Width camera 2
        bp1     Ball positions camera 1
        bp2     Ball positions camera 2
        pc1     Corner positions camera 1
        pc2     Corner positions camera 2
        c3d     3D corner positions
        P1      Camera matrix 1
        P2      Camera matrix 2
                Factorizations of camera matrices:
        K1
        K2
        A1
        A2
                Normalized camera matrices:
        P1norm
        P2norm
                Detected points strokes etc:
        points
        times
        bounces
    '''

    # Initiate and calculate cameras, points, etc.
    def __init__(self, height1,width1, height2,width2,corners1,corners2,ball_pos_1,ball_pos_2,fps):
        self.fps = fps
        self.h1 = height1
        self.w1 = width1
        self.h2 = height2
        self.w2 = width2
        self.bp1 = np.transpose(ball_pos_1)
        self.bp2 = np.transpose(ball_pos_2)
        # Points in cornersX should correspond: p1-p3, p2-p4, p3-p1, p4-p2
        self.pc1 = np.copy(corners1)
        self.pc2 = np.zeros([6,3])
        self.pc2[0,:] = corners2[2,:]
        self.pc2[1, :] = corners2[3, :]
        self.pc2[2, :] = corners2[0, :]
        self.pc2[3, :] = corners2[1, :]
        self.pc2[4, :] = corners2[5, :]
        self.pc2[5, :] = corners2[4, :]
        self.pc1 = np.transpose(self.pc1)
        self.pc2 = np.transpose(self.pc2)
        # Calculate camera matrices P1 and P2 from 6 known points
        p1 = [0,ytable,0,1]
        p2 = [xtable,ytable,0,1]
        p3 = [xtable,0,0,1]
        p4 = [0,0,0,1]
        p5 = [xtable / 2, -0.1525, 0.15, 1]
        p6 = [xtable / 2, ytable+0.1525, 0.15, 1]
        self.c3d = np.transpose(np.array([p1,p2,p3,p4,p5,p6]))
        # Calculate P1 and P2
        self.P1 = calc_P(self.c3d, self.pc1)
        self.P2 = calc_P(self.c3d, self.pc2)
        [r1,q1] = rq(self.P1)
        [r2,q2] = rq(self.P2)
        self.K1 = r1
        self.K2 = r2
        self.A1 = q1
        self.A2 = q2
        self.P1norm = np.matmul(np.linalg.inv(self.K1),self.P1)
        self.P2norm = np.matmul(np.linalg.inv(self.K2),self.P2)

        # Generate 3d points
        self.p3d = []
        for j in range(len(self.bp1)):
            if is_zero(self.bp1[j,:]) or is_zero(self.bp2[j,:]):
                self.p3d.append(np.array([0,0,1]))
            else:
                point = self.calc_3d_point(self.bp1[j,:],self.bp2[j,:])
                if inside_range(point):
                    self.p3d.append(point)
                else:
                    self.p3d.append(np.array([0,0,0]))
        self.p3d = np.array(self.p3d)

        # Remove outliers
        samecount = 0
        for i in range(np.size(self.p3d,0)-4):
            neighs = []
            for j in range(5):
                if not is_zero(self.p3d[i+j,:]) and j!=2:
                    neighs.append(self.p3d[i+j,:])
            if len(neighs)>0:
                arr = np.array(neighs)
                means = np.mean(arr,axis=0)
                norm = np.linalg.norm(self.p3d[i+2,:]-means)
                if norm>0.5 and samecount<10:
                    samecount += 1
                    self.p3d[i+2,:] = 0
                else:
                    samecount = 0

        # Extract strokes in every point
        self.points,self.times = divide_into_points(self.p3d)

        # Interpolate between positions in each point
        for i in range(len(self.points)):
            initpos = 0
            curpos = initpos
            to_interpol = []
            times = []
            while True:
                if self.points[i][curpos,0]!=0:
                    to_interpol.append(self.points[i][curpos,:])
                    times.append(curpos)
                curpos += 1
                if curpos==self.points[i].shape[0]:
                    break
                if len(times)==4:
                    if times[3]-times[0]>3:
                        self.points[i][times[0]:times[3]+1,:] = interpolate_missing(to_interpol[0],to_interpol[1],to_interpol[2],to_interpol[3],
                                                                                    times[0],times[1],times[2],times[3])
                    initpos = times[3]-2
                    curpos = initpos
                    to_interpol = []
                    times = []
        # Put back into original data
        self.p3d = np.zeros(self.p3d.shape)
        for point,time in zip(self.points,self.times):
            self.p3d[time[0]:time[1],:] = point

        for i in range(len(self.points)):
            self.points[i] = divide_into_strokes(self.points[i])
        # Find bounces for every stroke
        self.bounces = find_bounces(self.points)

    # Finds a 3D point from 2 2D points
    def calc_3d_point(self,x1,x2):  # Points should be [a,b,1]
        x1norm = np.matmul(np.linalg.inv(self.K1),x1)
        x2norm = np.matmul(np.linalg.inv(self.K2),x2)
        M = np.zeros([6,6])
        M[0:3,0:4] = self.P1norm
        M[3:6,0:4] = self.P2norm
        M[0:3,4] = -x1norm
        M[3:6,5] = -x2norm
        [_,_,V] = np.linalg.svd(M)
        v = V[5,:]
        X = pflat(np.reshape(v[0:4],[4,1]))
        return np.reshape(X[0:3],[3,])

    # Plot trajectory from P1 or P2
    def plot_trajectory(self, camera):
        x = [0, 0, 2.74, 2.74, 0, 1.37, 1.37, 1.37, 1.37, 1.37, 1.37]
        y = [0, 1.525, 1.525, 0, 0, 0, -0.1525, -0.1525, 1.525 + 0.1525, 1.525 + 0.1525, -0.1525]
        z = [0, 0, 0, 0, 0, 0, 0, 0.1525, 0.1525, 0, 0]
        w = [1,1,1,1,1,1,1,1,1,1,1]
        data = np.array([x,y,z,w])
        P = self.P1
        bp = self.bp1
        h = self.h1
        w = self.w1
        if camera==2:
            P = self.P2
            bp = self.bp2
            h = self.h2
            w = self.w2
        data = pflat(np.matmul(P,data))
        plt.plot(data[0,:],data[1,:],'r')
        balldata = np.vstack((np.transpose(self.p3d),np.ones((1,self.p3d.shape[0]))))
        balldata = pflat(np.matmul(P,balldata))
        plt.scatter(balldata[0,:],balldata[1,:],c='b')
        #plt.plot(bp[:, 0], bp[:, 1], c='m')
        plt.xlim(0, w)
        plt.ylim(0, h)
        plt.show()

    # Plots path taken by ball in 3D in one point
    def plot_3d_point(self,pointnbr):
        fig = plt.figure()
        ax = Axes3D(fig)
        for stroke in self.points[pointnbr-1]:
            strokecopy = []
            for i in range(stroke.shape[0]):
                if stroke[i,0]!=0 and stroke[i,1]!=0:
                    strokecopy.append([stroke[i,0],stroke[i,1],stroke[i,2]])
            sc = np.array(strokecopy)
            ax.scatter([sc[i,0] for i in range(sc.shape[0])], [sc[i,1] for i in range(sc.shape[0])], [sc[i,2] for i in range(sc.shape[0])])
        x = [0,0,2.74,2.74,0,1.37,1.37,1.37,1.37,1.37,1.37]
        y = [0,1.525,1.525,0,0,0,-0.1525,-0.1525,1.525+0.1525,1.525+0.1525,-0.1525]
        z = [0,0,0,0,0,0,0,0.1525,0.1525,0,0]
        pos1 = -np.matmul(np.linalg.inv(self.A1[0:3,0:3]),self.A1[:,3])
        pos2 = -np.matmul(np.linalg.inv(self.A2[0:3, 0:3]), self.A2[:, 3])
        dir1 = self.A1[2,:]
        dir2 = self.A2[2, :]
        ax.scatter(pos1[0],pos1[1],pos1[2],c='k')
        ax.scatter(pos2[0], pos2[1], pos2[2], c='k')
        ax.quiver(pos1[0],pos1[1],pos1[2], dir1[0], dir1[1], dir1[2], length=1, normalize=True)
        ax.quiver(pos2[0], pos2[1], pos2[2], dir2[0], dir2[1], dir2[2], length=1, normalize=True)
        ax.plot(x,y,z,'r')
        # maxpos = max([np.max(pos1),np.max(pos2)])
        ax.set_xlim(-1, 4)
        ax.set_ylim(-2, 3)
        ax.set_zlim(-1, 3)
        plt.show()

    # Plots path taken by ball in 2D in one stroke
    def visualize_2d_strokes(self,pointnbr):
        for idx in range(len(self.points[pointnbr - 1])):
            stroke = self.points[pointnbr - 1][idx]
            if stroke.shape[0]>10:
                plt.xlim(-1, 3.74)
                plt.ylim(-1, 3)
                strokecopy = []
                for i in range(stroke.shape[0]):
                    if stroke[i, 0] != 0 and stroke[i, 1] != 0:
                        strokecopy.append([stroke[i, 0], stroke[i, 2]])
                sc = np.array(strokecopy)
                plt.scatter([sc[i, 0] for i in range(sc.shape[0])], [sc[i, 1] for i in range(sc.shape[0])])
                if self.bounces[pointnbr - 1][idx][0, 0]!=0 or self.bounces[pointnbr - 1][idx][0, 1]!=0:
                    plt.scatter([self.bounces[pointnbr - 1][idx][i, 0] for i in range(self.bounces[pointnbr - 1][idx].shape[0])],
                                [self.bounces[pointnbr - 1][idx][i, 2] for i in range(self.bounces[pointnbr - 1][idx].shape[0])])
                plt.plot([0,2.74,1.37,1.37],[0,0,0,0.15],'r')
                plt.show()
                idx += 1
                plt.close()

    # Plots path taken by ball in 3D in one stroke
    def visualize_3d_strokes(self, pointnbr):
        for idx in range(len(self.points[pointnbr - 1])):
            stroke = self.points[pointnbr - 1][idx]
            if stroke.shape[0]>10:
                fig = plt.figure()
                ax = Axes3D(fig)
                x = [0, 0, 2.74, 2.74, 0, 1.37, 1.37, 1.37, 1.37, 1.37, 1.37]
                y = [0, 1.525, 1.525, 0, 0, 0, -0.1525, -0.1525, 1.525 + 0.1525, 1.525 + 0.1525, -0.1525]
                z = [0, 0, 0, 0, 0, 0, 0, 0.1525, 0.1525, 0, 0]
                pos1 = -np.matmul(np.linalg.inv(self.A1[0:3, 0:3]), self.A1[:, 3])
                pos2 = -np.matmul(np.linalg.inv(self.A2[0:3, 0:3]), self.A2[:, 3])
                dir1 = self.A1[2, :]
                dir2 = self.A2[2, :]
                ax.scatter(pos1[0], pos1[1], pos1[2], c='k')
                ax.scatter(pos2[0], pos2[1], pos2[2], c='k')
                ax.quiver(pos1[0], pos1[1], pos1[2], dir1[0], dir1[1], dir1[2], length=1, normalize=True)
                ax.quiver(pos2[0], pos2[1], pos2[2], dir2[0], dir2[1], dir2[2], length=1, normalize=True)
                ax.plot(x, y, z, 'r')
                # maxpos = max([np.max(pos1),np.max(pos2)])
                ax.set_xlim(-1, 4)
                ax.set_ylim(-2, 3)
                ax.set_zlim(-1, 3)
                strokecopy = []
                for i in range(stroke.shape[0]):
                    if stroke[i, 0] != 0 and stroke[i, 1] != 0:
                        strokecopy.append([stroke[i, 0], stroke[i, 1], stroke[i, 2]])
                sc = np.array(strokecopy)
                ax.scatter([sc[i, 0] for i in range(sc.shape[0])], [sc[i, 1] for i in range(sc.shape[0])], [sc[i, 2] for i in range(sc.shape[0])])
                if self.bounces[pointnbr - 1][idx][0, 0] != 0 or self.bounces[pointnbr - 1][idx][0, 1] != 0:
                    ax.scatter([self.bounces[pointnbr-1][idx][i,0] for i in range(self.bounces[pointnbr-1][idx].shape[0])],
                                [self.bounces[pointnbr-1][idx][i,1] for i in range(self.bounces[pointnbr-1][idx].shape[0])],
                                [self.bounces[pointnbr-1][idx][i,2] for i in range(self.bounces[pointnbr-1][idx].shape[0])])
                plt.show()
                idx += 1
                plt.close()

    # Generate animation of path in 3D
    def animate_3d_path(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        x = [0, 0, 2.74, 2.74, 0, 1.37, 1.37, 1.37, 1.37, 1.37, 1.37]
        y = [0, 1.525, 1.525, 0, 0, 0, -0.1525, -0.1525, 1.525 + 0.1525, 1.525 + 0.1525, -0.1525]
        z = [0, 0, 0, 0, 0, 0, 0, 0.1525, 0.1525, 0, 0]
        #pos1 = -np.matmul(np.linalg.inv(self.A1[0:3, 0:3]), self.A1[:, 3])
        #pos2 = -np.matmul(np.linalg.inv(self.A2[0:3, 0:3]), self.A2[:, 3])
        #dir1 = self.A1[2, :]
        #dir2 = self.A2[2, :]
        #ax.scatter(pos1[0], pos1[1], pos1[2], c='k')
        #ax.scatter(pos2[0], pos2[1], pos2[2], c='k')
        #ax.quiver(pos1[0], pos1[1], pos1[2], dir1[0], dir1[1], dir1[2], length=1, normalize=True)
        #ax.quiver(pos2[0], pos2[1], pos2[2], dir2[0], dir2[1], dir2[2], length=1, normalize=True)
        ax.plot(x, y, z, 'b', linewidth=2)
        # # Bounces
        # xdata = []
        # ydata = []
        # zdata = []
        # for bounceid in self.bounces:
        #     xdata.append(self.p3d[bounceid, 0])
        #     ydata.append(self.p3d[bounceid, 1])
        #     zdata.append(0)
        # ax.scatter(xdata, ydata, zdata)
        # maxpos = max([np.max(pos1),np.max(pos2)])
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-1, 3)
        ax.set_zlim(-1, 1.5)

        data = np.transpose(self.p3d)
        line = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1], linestyle="-",marker='.',c='r',linewidth=1)[0]

        def update_points(num,data,line):
            if num>20:
                line.set_data(data[0:2, num-10:num])
                line.set_3d_properties(data[2, num-10:num])
            else:
                line.set_data(data[0:2,:num])
                line.set_3d_properties(data[2,:num])
            return line

        anim = animation.FuncAnimation(fig, update_points, frames=len(self.p3d), fargs=(data,line), interval=0, blit=False)
        plt.show()

    # Calculate velocity during flight
    def calc_velocity(self,plotting=False): #### DEPRECATED
        dt = 1.0/self.fps
        vel = []
        time = [i+3 for i in range(len(self.p3d)-4)]
        time = [float(nbr)/self.fps for nbr in time]
        for i in range(len(self.p3d)-4):
            ok = True
            dx = 0
            dy = 0
            dz = 0
            for j in range(4):
                if is_zero(self.p3d[i+j]):
                    ok = False
                else:
                    dx += np.abs(self.p3d[i+j+1][0]-self.p3d[i+j][0])
                    dy += np.abs(self.p3d[i+j+1][1]-self.p3d[i+j][1])
                    dz += np.abs(self.p3d[i+j+1][2]-self.p3d[i+j][2])
            if ok:
                length = np.sqrt(dx**2+dy**2+dz**2)
                vel.append(length/5/dt)
            else:
                vel.append(0)
        if plotting:
            plt.plot(time[:],vel[:])
            plt.show()
        return vel

    # Ideally plots the points of contact in 3D
    def plot_turning_points(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        xdata = [self.p3d[i][0] for i in self.turns]
        ydata = [self.p3d[i][1] for i in self.turns]
        zdata = [self.p3d[i][2] for i in self.turns]
        ax.scatter(xdata,ydata,zdata)
        x = [0, 0, 2.74, 2.74, 0, 1.37, 1.37, 1.37, 1.37, 1.37, 1.37]
        y = [0, 1.525, 1.525, 0, 0, 0, -0.1525, -0.1525, 1.525 + 0.1525, 1.525 + 0.1525, -0.1525]
        z = [0, 0, 0, 0, 0, 0, 0, 0.1525, 0.1525, 0, 0]
        ax.plot(x, y, z, 'r')
        # maxpos = max([np.max(pos1),np.max(pos2)])
        ax.set_xlim(-1, 4)
        ax.set_ylim(-2, 3)
        ax.set_zlim(-1, 3)
        plt.show()

    # Plot a heatmap of all bounces on the table
    def bounce_heatmap(self):
        x = []
        y = []
        for point in self.bounces:
            for stroke in point:
                for i in range(stroke.shape[0]):
                    if stroke[i,0]!=0 or stroke[i,1]!=0:
                        x.append(stroke[i,0])
                        y.append(stroke[i,1])
        bins = 1000
        s = 32
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins,range=[[0,2.74],[0,1.525]])
        heatmap = gaussian_filter(heatmap, sigma=s)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=cm.jet)
        plt.plot([1.37,1.37],[0,1.525],'r')
        plt.show()

    # Set the ball positions to new values
    def set_ball_trace(self,bp1,bp2):
        self.bp1 = np.transpose(bp1)
        self.bp2 = np.transpose(bp2)
        if self.bp1.shape[0]>self.bp2.shape[0]:
            for i in range(self.bp1.shape[0]-self.bp2.shape[0]):
                self.bp2 = np.append(self.bp2,np.array([[0,0,1]]),axis=0)
        elif self.bp1.shape[0]<self.bp2.shape[0]:
            for i in range(-self.bp1.shape[0]+self.bp2.shape[0]):
                self.bp1 = np.append(self.bp1,np.array([[0,0,1]]),axis=0)

# RQ-factorization
def rq(a):
    [m,n] = a.shape
    e = np.eye(m)
    p = np.fliplr(e)
    [q0,r0] = qr(np.matmul(p,np.matmul(np.transpose(a[:,0:m]),p)))
    r = np.matmul(p,np.matmul(np.transpose(r0),p))
    q = np.matmul(p, np.matmul(np.transpose(q0), p))
    fix = np.diag(np.sign(np.diag(r)))
    r = np.matmul(r,fix)
    q = np.matmul(fix,q)
    if n>m:
        q = np.concatenate((q,np.matmul(np.linalg.inv(r),a[:,m:n])),axis=1)
    return r,q

# Pointwise division with last coordinate
def pflat(x):
    y = np.copy(x)
    for i in range(x.shape[1]):
        y[:,i] = y[:,i]/y[x.shape[0]-1,i]
    return y

# Create artificial ball movement
def create_trace(P1,P2):
    bp3d1 = [[x/33-0.12,1.525-x**2/10000,0.3*abs(math.cos(2*math.pi*x/100)),1] for x in range(100)]
    bp3d2 = [[bp3d1[len(bp3d1)-1][0]-0.01-x/33,bp3d1[len(bp3d1)-1][1], 0.3 * abs(math.cos(2 * math.pi * x / 350)),1] for x in range(100)]
    bp3d = bp3d1 + bp3d2
    bp3d = np.transpose(np.array(bp3d))
    p1 = pflat(np.matmul(P1,bp3d))
    p2 = pflat(np.matmul(P2,bp3d))
    return np.transpose(p1),np.transpose(p2)

# Create artificial table position
def table_position(height,width):
    add1 = 0
    add2 = 40
    p1 = [[2*width / 10, height*4/10, 1],
          [8*width/10, height*4/10, 1],
          [9 * width / 10, height / 10, 1],
          [1 * width / 10, height / 10, 1],
          [5 * width / 10, height / 10+add1, 1],
          [5 * width / 10, height *4/ 10+add2, 1]]
    p2 = [[480,560, 1],
          [1040,360, 1],
          [800, 80, 1],
          [160,360, 1],
          [420,260, 1],
          [764, 540, 1]]
    return np.array(p1), np.array(p2)

# Find turns made by ball, returns all indexes in ball-position vectors
def find_turns(points):
    turns = [0]
    prevdir = np.sign(points[1,0]-points[0,0])
    for i in range(points.shape[0]-1):
        if points[i,0]!=0 and points[i+1,0]!=0:
            x0 = points[i,0]
            x1 = points[i+1,0]
            dir = np.sign(x1-x0)
            if dir==-prevdir and dir!=0:
                turns.append(i)
                prevdir = dir
            elif prevdir==0:
                prevdir=dir
    turns.append(points.shape[0]-1)
    return turns

# Calculates camera matrix from a set of 6 point correspondences
def calc_P(p3d,p2d):
    npoints = p2d.shape[1]
    mean = np.mean(p2d,1)
    std = np.std(p2d,axis=1)
    N = np.array([[1/std[0],0,-mean[0]/std[0]],
                  [0,1/std[1],-mean[1]/std[1]],
                  [0,0,1]])
    p2dnorm = np.matmul(N,p2d)
    M = np.zeros([3*npoints,12+npoints])
    for i in range(npoints):
        M[3*i,0:4] = p3d[:,i]
        M[3*i+1,4:8] = p3d[:,i]
        M[3*i+2,8:12] = p3d[:,i]
        M[3*i:3*i+3,12+i] = -p2dnorm[:,i]
    [U,S,V] = svd(M)
    v = V[V.shape[0]-1,:]
    P = np.reshape(v[0:12],[3,4])
    testsign = np.matmul(P,p3d[:,1])
    if testsign[2]<0:
        P = -P
        print('changed sign of P')
    P = np.matmul(np.linalg.inv(N),P)
    return P

# Checks if point is zero and should be ignored
def is_zero(p):
    if p[0]==0 and p[1]==0:
        return True
    else:
        return False

# Checks if point is within reasonable range from table
def inside_range(point):
    return -1<point[0]<3.74 and -1<point[1]<2.525 and -1<point[2]<3

# Divides 3d positions into tt-points based on existence of ball
def divide_into_points(p3d):
    ballfound = np.zeros(p3d.shape[0])
    ballfound[p3d[:, 0] != 0] = 1
    ma = np.zeros(p3d.shape[0])
    kern = 30
    thresh = 0.5
    idxs = [0]
    for i in range(p3d.shape[0]):
        vec = ballfound[max(0, i - kern):min(p3d.shape[0] - 1, i + kern)]
        ma[i] = sum(vec) / vec.shape[0]
        if i>0 and (ma[i-1]<=thresh<ma[i] or ma[i-1]>thresh>=ma[i]):
            idxs.append(i)
    idxs.append(p3d.shape[0])
    points = np.vsplit(p3d,idxs)
    if ma[0]<thresh:
        toremove = 0
    else:
        toremove = 1
    actualpoints = []
    actualtimes = []
    for i in range(len(points)):
        if (i+toremove)%2==0:
            removedbefore = 0
            removedafter = 0
            for j in range(points[i].shape[0]):
                if points[i][j,0]==0:
                    removedbefore += 1
                else:
                    break
            for j in range(points[i].shape[0]):
                if points[i][-(1+j), 0] == 0:
                    removedafter += 1
                else:
                    break
            if removedafter>0:
                toadd = points[i][removedbefore:-removedafter,:]
            else:
                toadd = points[i][removedbefore:,:]
            if sum(toadd[:,0]>0)>3:
                actualpoints.append(toadd)
                actualtimes.append([idxs[i-1]+removedbefore,idxs[i]-removedafter])
    # plt.plot(range(ballfound.shape[0]), ballfound)
    # plt.plot(range(ballfound.shape[0]), ma)
    # plt.show()
    return actualpoints,actualtimes

# Divides a point into a number of strokes based on x-direction of ball
def divide_into_strokes(points):
    strokes = []
    turns = find_turns(points)
    for i in range(len(turns)-1):
        strokes.append(points[turns[i]:turns[i+1]])
    return strokes

# Finds the bounce(s) for each stroke
def find_bounces(points):
    bounces = []
    for point in points:
        point_bounces = []
        shots_made = 0
        for stroke in point:
            if stroke.shape[0] > 10:
                z = stroke[:, 2]
                minimas = argrelextrema(z, np.less)[0]
                if len(minimas) == 0:
                    point_bounces.append(np.reshape(np.array([0, 0, 0]), [1, 3]))
                elif shots_made == 0:
                    values = z[minimas]
                    argmin1 = np.argmin(values)
                    values[argmin1] = 5
                    argmin2 = np.argmin(values)
                    args = [minimas[argmin1], minimas[argmin2]]
                    bounce = np.vstack((stroke[min(args), :], stroke[max(args), :]))
                    bounce[0, 2] = 0
                    bounce[1, 2] = 0
                    point_bounces.append(bounce)
                    shots_made += 1
                else:
                    values = z[minimas]
                    argmin = np.argmin(values)
                    bounce = np.reshape(stroke[minimas[argmin], :], [1, 3])
                    bounce[0, 2] = 0
                    point_bounces.append(bounce)
            else:
                point_bounces.append(np.reshape(np.array([0, 0, 0]), [1, 3]))
        bounces.append(point_bounces)
    return bounces

# Interpolate positions of missing points
def interpolate_missing(a,b,c,d,t0,t1,t2,t3):
    matinv = np.linalg.inv([[1,t0,t0**2,t0**3],
                        [1,t1,t1**2,t1**3],
                        [1,t2,t2**2,t2**3],
                        [1,t3,t3**2,t3**3]])
    coeff = np.zeros([3,4])
    for i in range(3):
        values = np.array([a[i],b[i],c[i],d[i]])
        coeff[i,:] = matinv@values
    missing = np.zeros([t3-t0+1,3])
    for i in range(missing.shape[0]):
        missing[i,:] = coeff@np.array([1,t0+i,(t0+i)**2,(t0+i)**3])
    return missing

# h = 700
# w = 1280
# c1, c2 = table_position(h, w)     # Should come from ML-algorithm
# a = analyzer(h,w,h,w,c1,c2,[],[])
# [P1,P2] = a.get_cameras()
# bp1, bp2 = create_trace(P1,P2) # Should come from ML-algorithm
# bp1 = np.append(bp1,np.array([[0,0,1]]),axis=0)
# a.set_ball_trace(np.transpose(bp1),np.transpose(bp2))
# str = a.gen_3d_strokes()
# a.plot_3d_path(str)
# a.plot_trajectory_2()
