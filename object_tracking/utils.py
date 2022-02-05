import cv2
import torch
import numpy as np


def save_video(frames, video_name):
    """
    This function save frames as a video
    :param frames:
    :param video_name:
    :return:
    """

    # Initialize the output video file
    file_path = 'output_data/' + video_name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(file_path, fourcc, 25, (320, 240))

    # Write video
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
