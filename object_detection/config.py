
class Config:
    WEIGHTS = './object_detection/models/yolov5m6.pt'  # model.pt path(s)
    SOURCE = 'data/videos'  # file/folder, 0 for webcam
    BATCH_OF_FRAMES = 1
    IMAGE_SIZE = 640  # Inference size (pixels)
    CONFIDENCE_THRESHOLD = 0.8  # Object confidence threshold
    IOU_THRESHOLD = 0.45  # IOU threshold for NMS
    MAX_DETECTION = 1000  # Maximum number of detections per image
    DEVICE = ''  # Cuda device, i.e. 0 or 0,1,2,3 or cpu
    VIEW_IMAGE = False  # Display results
    SAVE_TEXT = False  # Save results to *.txt
    SAVE_CONFIDENCE = False  # Save confidences in --save-txt labels
    SAVE_CROP = False  # Save cropped prediction boxes
    NO_SAVE = False  # Do not save images/videos
    CLASSES = None  # Filter by class: --class 0, or --class 0 2 3
    AGNOSTIC_NMS = False  # Class-agnostic NMS
    AUGMENT = False  # Augmented inference
    UPDATE = False  # Update all models'
    PROJECT = 'runs/detect'  # Save results to project/name
    NAME = 'exp'  # Save results to project/name
    EXIST_OK = False  # Existing project/name ok, do not increment'
    LINE_THICKNESS = 3  # Bounding box thickness (pixels)
    HIDE_LABELS = False  # Hide labels
    HIDE_CONFIDENCE = False  # hide confidences
