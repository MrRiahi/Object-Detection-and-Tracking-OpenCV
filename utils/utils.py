import os
import cv2
import numpy as np


class UtilityFunctions:

    @staticmethod
    def get_high_probability_object_info(predictions, object_names):
        """
        This method gets the bounding box and class name of object with the most probability.
        :param predictions: list of predictions result
        :param object_names: list of object names
        :return:
        """

        max_index = predictions[0][:, 4].argmax()

        prediction = predictions[0][max_index]

        bounding_box = UtilityFunctions.get_bounding_box(prediction=prediction)
        bounding_box = bounding_box.astype('uint16')

        object_name = UtilityFunctions.get_object_name(prediction=prediction, object_names=object_names)

        return bounding_box, object_name

    @staticmethod
    def get_bounding_box(prediction):
        """
        This method gets the bounding box of the object
        :param prediction:
        :return:
        """

        bounding_box = prediction[:4]

        return bounding_box

    @staticmethod
    def get_object_name(prediction, object_names):
        """
        This method gets the object name of prediction
        :param prediction:
        :param object_names:
        :return:
        """

        object_name = object_names[int(prediction[-1])]

        return object_name

    @staticmethod
    def get_iou(bounding_box, bounding_boxes):
        """
        This method gets the IOU of bounding_box with every bounding box in bounding_boxes.
        :param bounding_box:
        :param bounding_boxes:
        :return:
        """

        iou_list = []
        for bbox in bounding_boxes:
            iou_value = UtilityFunctions.calculate_iou(bounding_box_1=bounding_box, bounding_box_2=bbox)
            iou_list.append(iou_value)

        return iou_list

    @staticmethod
    def calculate_iou(bounding_box_1, bounding_box_2):
        """
        This method calculates the iou of bounding_box_1 and bounding_box_2
        :param bounding_box_1:
        :param bounding_box_2:
        :return:
        """

        # Get the coordinate of the intersection part
        x_top_left = max(bounding_box_1[0], bounding_box_2[0])
        y_top_left = max(bounding_box_1[1], bounding_box_2[1])
        x_bottom_right = min(bounding_box_1[2], bounding_box_2[2])
        y_bottom_right = min(bounding_box_1[3], bounding_box_2[3])

        # Compute the ares of bounding_box_1, bounding_box_2, and intersection
        area_1 = (bounding_box_1[2] - bounding_box_1[0]) * (bounding_box_1[3] - bounding_box_1[1])
        area_2 = (bounding_box_2[2] - bounding_box_2[0]) * (bounding_box_2[3] - bounding_box_2[1])
        intersection = (x_bottom_right - x_top_left) * (y_bottom_right - y_top_left)

        if intersection <= 0:
            print('Tracker Failed!')
            iou = None
        else:
            iou = intersection / float(area_1 + area_2 - intersection)

        return iou

    @staticmethod
    def find_correct_bounding_box(iou_list, predictions, object_names, threshold=0.1):
        """
        This method find the correct bounding box based on IOU values. This consists of three steps:
        1) Thresholding on the IOU list
        2) Get the most similar IOU
        3) Extract the bounding box of the most similar IOU and name of the object
        :param iou_list:
        :param predictions:
        :param object_names:
        :param threshold:
        :return:
        """

        # Thresholding
        iou_threshold = np.greater(iou_list, threshold)

        # Find the index of maximum value of IOU
        index = -1
        if any(iou_threshold):
            index = np.argmax(iou_list)

        # Extract the bounding box and object name
        bounding_box = []
        object_name = ''
        if index != -1:
            prediction = predictions[0][index]

            bounding_box = UtilityFunctions.get_bounding_box(prediction=prediction)
            bounding_box = bounding_box.astype('uint16')

            object_name = UtilityFunctions.get_object_name(prediction=prediction, object_names=object_names)

        return bounding_box, object_name

    @staticmethod
    def save_video(frames, video_name):
        """
        This function save frames as a video
        :param frames:
        :param video_name:
        :return:
        """

        # Initialize the output video file
        os.makedirs('output_data', exist_ok=True)
        file_path = 'output_data/' + video_name
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'h264')
        video_writer = cv2.VideoWriter(file_path, fourcc, 25, (428, 234))

        # Write video
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()
