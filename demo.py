import numpy as np
import analysis_functions
import argparse
from pathlib import Path

OPTIONS = [
    "visualize_3d_strokes",
    "visualize_2d_strokes",
    "animate_3d_path",
    "bounce_heatmap",
    "plot_3d_point",
    "plot_trajectory",
]


def main(args):
    """
    Creates an "analyzer" object for visualization of processed video data.
    """
    if not args.viztype in OPTIONS:
        print("Invalid vizualization chosen, choose from: ", str(OPTIONS))
        raise

    vidname = args.video_name

    c1 = np.load(
        str(
            Path.joinpath(
                Path(__file__).parents[0].resolve(), "data/" + vidname + "/c1.npy"
            )
        )
    )
    c2 = np.load(
        str(
            Path.joinpath(
                Path(__file__).parents[0].resolve(), "data/" + vidname + "/c2.npy"
            )
        )
    )

    ball_pos1 = np.load(
        str(
            Path.joinpath(
                Path(__file__).parents[0].resolve(),
                "data/" + vidname + "/ballpath1.npy",
            )
        )
    )
    param1 = np.load(
        str(
            Path.joinpath(
                Path(__file__).parents[0].resolve(), "data/" + vidname + "/param1.npy"
            )
        )
    )
    height1 = param1[0]
    width1 = param1[1]
    fps1 = param1[2]

    ball_pos2 = np.load(
        str(
            Path.joinpath(
                Path(__file__).parents[0].resolve(),
                "data/" + vidname + "/ballpath2.npy",
            )
        )
    )
    param2 = np.load(
        str(
            Path.joinpath(
                Path(__file__).parents[0].resolve(), "data/" + vidname + "/param2.npy"
            )
        )
    )
    height2 = param2[0]
    width2 = param2[1]
    fps2 = param2[2]

    analyzer = analysis_functions.analyzer(
        height1,
        width1,
        height2,
        width2,
        c1,
        c2,
        np.transpose(ball_pos1),
        np.transpose(ball_pos2),
        fps1,
    )

    # Available functions to run:
    if args.viztype == "visualize_3d_strokes":
        analyzer.visualize_3d_strokes(
            args.pnbr
        )  # Plots one stroke at a time in 3d in point nbr pnbr
    elif args.viztype == "visualize_2d_strokes":
        analyzer.visualize_2d_strokes(
            args.pnbr
        )  # Plots one stroke at a time in 2d in point nbr pnbr
    elif args.viztype == "animate_3d_path":
        analyzer.animate_3d_path()  # Animates the 3d path taken in the whole video
    elif args.viztype == "bounce_heatmap":
        analyzer.bounce_heatmap()  # Plots a heatmap of the detected bounces
    elif args.viztype == "plot_3d_point":
        analyzer.plot_3d_point(args.pnbr)  # Plots the trajectory of point pnbr
    else:
        analyzer.plot_trajectory(
            args.pnbr
        )  # Plots all trajectories from camera 1/2 perspective


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_name",
        type=str,
        required=True,
        help="Name of video used in processing.",
    )
    parser.add_argument(
        "--viztype",
        type=str,
        required=True,
        help="Type of visualization to run.",
    )
    parser.add_argument(
        "--pnbr",
        type=int,
        required=False,
        default=1,
        help="The ID of the point to analyze in the video.",
    )

    args = parser.parse_args()
    main(args)
