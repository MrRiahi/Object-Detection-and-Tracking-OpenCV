import cv2
import time
import numpy as np

from config import Config as Cfg
from utils.utils import UtilityFunctions
from object_tracking.track import get_tracker
from object_detection.detect import ObjectDetection, detect

# Create a VideoCapture object
video_name = 'test_1.mp4'
cap = cv2.VideoCapture(f'{Cfg.SOURCE}/{video_name}')

# Load and initialize model
object_detector = ObjectDetection()

# Initialize tracker
tracker = get_tracker()

# Check whether the video open successfully or not
if not cap.isOpened():
    raise Exception('Error opening video stream or file')

# Initialize parameters
cnt = 0
is_track = False
ret = False
frames = []
bounding_box = []
times = []

# Read until video is completed
while cap.isOpened():
    print(f'{cnt}/')
    is_empty, frame = cap.read()

    if not is_empty:
        print('The video stream is finish')
        break

    tic = time.time()
    # detect()
    if cnt % Cfg.BATCH_OF_FRAMES == 0:
        # Use object detection model to update bounding box and detect new objects
        predictions = object_detector.detect_objects(frames=[frame])
        print(f'predictions::{predictions}')

        # Get bounding box
        if len(predictions):
            if len(bounding_box):
                # Get IOU between bounding_box and prediction
                iou_list = UtilityFunctions.get_iou(bounding_box=bounding_box, bounding_boxes=predictions[0])

                # Find the correct bounding box based on IOU values
                bounding_box, object_name = UtilityFunctions.find_correct_bounding_box(
                    iou_list=iou_list, predictions=predictions, object_names=object_detector.names)

            else:
                bounding_box, object_name = UtilityFunctions.get_high_probability_object_info(
                    predictions=predictions,
                    object_names=object_detector.names)

            # Update tracker with new bounding box
            if len(bounding_box):
                tracker.init(frame=frame, bbox=bounding_box)
                is_plot, bbox = tracker.update(frame=frame, predictions=predictions)

                if len(bbox) > 0:
                    bounding_box = bbox

                is_track = True
                ret = True

            else:
                ret = False
                is_track = False

        else:
            ret = False

    else:

        if is_track:
            # Use object tracker to track object
            is_plot, bbox = tracker.update(frame=frame, predictions=predictions)
            print('-------bbox', bbox)

            if len(bbox) > 0:
                predictions = tra
            #
            # ret, bounding_box = tracker.update(frame=frame)
            #
            # bounding_box = np.array([bounding_box[0], bounding_box[1],
            #                          bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]],
            #                         dtype='uint16')

    toc = time.time()
    times.append(toc-tic)
    cv2.putText(frame, str(round(1000 * (toc - tic), 2)) + ' ms', (0, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Draw bounding box
    # if ret:
    #     p1 = (bounding_box[0], bounding_box[1])
    #     p2 = (bounding_box[2], bounding_box[3])
    #
    #     cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    frames.append(frame)
    cnt += 1

# When everything done, release the video capture object
UtilityFunctions.save_video(frames=frames, video_name='test_1_DeepSort.mp4')
cap.release()
cv2.destroyAllWindows()

print(sum(times) / len(times))
