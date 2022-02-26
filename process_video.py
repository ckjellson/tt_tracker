import cv2
import numpy as np
import time
import argparse

"""
Functions for ball detection in each frame of a video
"""
cv2.setUseOptimized(True)
test = False  # Used during testing
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=15, varThreshold=50, detectShadows=False
)
kernel = np.ones((2, 2), np.uint8)
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 50
# Filter by Color.
params.filterByColor = True
params.blobColor = 255
# Filter by Area.
params.filterByArea = True
params.minArea = 30
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.75
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.9
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.08
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)


def read_video(path, flipped):
    """Main function

    Args:
        path (str):         path to video
        flipped (bool):     true if a video is read in upside-down
    Returns:
        height (int):           height of video frames
        width (int):            width of video frames
        ball_pos (np.array):    detected positions of balls in each frame
        fps (int):              frames per second
    """

    cap = cv2.VideoCapture(path)
    nbr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    height, width, ball_pos = track_ball(cap, nbr_frames, flipped)

    cap.release()
    cv2.destroyAllWindows()

    for i in range(ball_pos.shape[0]):
        ball_pos[i, 1] = height - ball_pos[i, 1] - 1

    return height, width, ball_pos, fps


def track_ball(cap, nbr_frames: int, flipped: bool) -> tuple:
    """Estimates trace of ball in video.

    Args:
        cap (_type_): Video object
        nbr_frames (int): Number of frames
        flipped (bool): true if video flipped

    Returns:
        tuple: video height,width and detected ball positions
    """
    ball_pos = np.ones([nbr_frames, 3])
    height = 0
    width = 0
    # Iterate through frames
    for i in range(nbr_frames):
        ret, frame = cap.read()
        if flipped:
            frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1)
        if i == 0:
            height, width, channels = frame.shape
        if ret:
            if width > 1280:
                frame = cv2.resize(
                    frame,
                    (int(frame.shape[1] / 2), int(frame.shape[0] / 2)),
                    interpolation=cv2.INTER_AREA,
                )
                ball_pos[i, 0:2] = find_ball(frame, height, width) * 2
            else:
                ball_pos[i, 0:2] = find_ball(frame, height, width)
        if i % 100 == 0:
            print(str(i) + " / " + str(nbr_frames))
    return height, width, ball_pos


def find_ball(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """Finds ball position in a frame.

    Args:
        frame (np.ndarray): Current frame.
        height (int): Video height.
        width (int): Video width.

    Returns:
        np.ndarray: 2D position of detected ball ([0,0] if none detected)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # t1 = time.time()
    gray[cv2.medianBlur(fgbg.apply(frame), ksize=5) == 0] = 0
    # t2 = time.time()
    keypoints = detector.detect(gray)
    # t3 = time.time()
    # print('fgbg:'+str(t2-t1))
    # print('detector:'+str(t3-t2))
    if test:
        im_with_keypoints = cv2.drawKeypoints(
            gray,
            keypoints,
            np.array([]),
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        cv2.imshow("", im_with_keypoints)
        cv2.waitKey()
    col = 0
    row = 0
    if len(keypoints) > 0:
        maxval = 0
        for i in range(len(keypoints)):
            x = int(keypoints[i].pt[0])
            y = int(keypoints[i].pt[1])
            val = np.sum(
                gray[
                    max([y - 3, 0]) : min([y + 3, height - 1]),
                    max([x - 3, 0]) : min([x + 3, width - 1]),
                ]
            )
            if val > maxval:
                col = x
                row = y
                maxval = val
    pos = np.array([col, row])
    if test:
        framecopy = np.copy(frame)
        cv2.circle(framecopy, (col, row), 10, color=(0, 255, 0), thickness=4)
        cv2.imshow("gray", framecopy)
        cv2.waitKey()
    return pos


if __name__ == "__main__":
    """This file is only run during testing"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=False,
        default="videos/outside2.mp4",
        help="path to a video for testing.",
    )
    parser.add_argument(
        "--flipped",
        type=str,
        required=False,
        default=False,
        help="Is the video flipped?",
    )

    args = parser.parse_args()
    height, width, ballpos, fps = read_video(args.path, args.flipped)
