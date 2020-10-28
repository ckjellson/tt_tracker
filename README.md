# tt_tracker

Tracking a table tennis ball in 3D using two cameras, and analyzing the result. Uses OpenCV and computer vision techniques to identify points, strokes, bounces, etc.

This project is not optimized for extended use, but rather showcases an idea for how OpenCV and computer vision can in a simple way be applied to analyze table tennis games.

## Demo

Uncomment a desired function furthest down in demo.py and run the script. This will create a "analyzer"-object. The following functions applied to this object are currently supported:

### visualize_3d_strokes()

Plots the path of one detected stroke at a time in 3d, detected bounce positions, and the camera positions.

![3dstroke](https://user-images.githubusercontent.com/37980849/97466582-8fbd8e80-1943-11eb-95f2-41e10e0acb18.PNG)

### visualize_2d_strokes()

Plots the path of one detected stroke at a time in 2d and detected bounce positions.

![2dstroke](https://user-images.githubusercontent.com/37980849/97466681-abc13000-1943-11eb-9ff4-0d739da60e8c.PNG)

### animate_3d_path()

Creates an animation of the estimated 3d path taken in the whole video.

![animation](https://user-images.githubusercontent.com/37980849/97466900-e9be5400-1943-11eb-92a5-4ab7db1f286a.PNG)

### bounce_heatmap()

Plots a heatmap of the detected bounces on the table.

![bounce_heatmap](https://user-images.githubusercontent.com/37980849/97466989-02c70500-1944-11eb-81b4-a3225d80edf3.PNG)

### plot_3d_point()

Plots the estimated path of a whole point in 3d together with camera positions.

![3dpoint](https://user-images.githubusercontent.com/37980849/97467107-2427f100-1944-11eb-9f0e-280c2f3e1af1.PNG)

### plot_trajectory()

Plots the estimated trajectory viewed from the perspective of one of the cameras.

![allpointscam](https://user-images.githubusercontent.com/37980849/97467284-546f8f80-1944-11eb-8cf7-b4b934337f2c.PNG)

## Usage

### Step 1: Clone repository

Clone repository to desired location.

### Step 2: Capture and save

Capture two videos simultaneously using two cameras from either side of the table.

Create a new folder called "videos_original" in the working directory and store the two videos here.

### Step 3: Crop (crop_video.py)

Create a new folder called "videos" in the working directory. Assign the variable vidname in crop_video.py a name of the instance that will be used to store the data later. Then run the script.

The function will open a window which shows the current frame from each camera. Press 1 or 2 to run a camera forward. When a synced position in the video has been reached, press esc. and the script will save the ramainder of the videos until one has no more frames into the "videos"-folder. This ensures equal length of the two videos.

### Step 4: Process videos (balltracker.py)

Set vidname to the chosen instance name in balltracker.py and then run the script. This will prompt you to locate corners and net positions in both videos. In the window that appears, double-click on the corner points and net positions in the order they appear here: (i.e. clock-wise from the top left to the bottom left, and then the top of the net, from bottom to top)

![camera1_corner](https://user-images.githubusercontent.com/37980849/97461066-aa8d0480-193d-11eb-9815-7282e87ac035.PNG)

After doing this the videos will be processed, and the detected position in each frame will be saved in the "data"-folder under the name of the video, together with the corner positions, fps, etc. This will take some time, depending on the length, resolution, etc. of the videos.

### Step 5: Analyze result (demo.py)

In demo.py set vidname to the name of your instance, and then choose to run the desired function.
