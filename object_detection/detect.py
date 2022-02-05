import cv2
import time
import torch
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn

from object_detection.config import Config as Cfg
from object_detection.models.experimental import attempt_load
from object_detection.utils.plots import colors, plot_one_box
from object_detection.utils.datasets import LoadStreams, LoadImages, letterbox
from object_detection.utils.torch_utils import select_device, load_classifier, time_synchronized
from object_detection.utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
                                           scale_coords, xyxy2xywh, set_logging, increment_path, save_one_box


class ObjectDetection:

    def __init__(self):

        # Get the model input image size
        self.image_size = Cfg.IMAGE_SIZE

        # Initialize
        set_logging()
        self.device = select_device(Cfg.DEVICE)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self._load_model()

        # Get model information
        self._get_info()

    def _load_model(self):
        """
        This method loads the model.
        :return:
        """

        # Load model
        self.model = attempt_load(Cfg.WEIGHTS, map_location=self.device)  # load FP32 model

        if self.half:
            self.model.half()  # to FP16

    def _get_info(self):
        """
        This method gets the information of the model
        :return:
        """

        self.stride = int(self.model.stride.max())  # model stride
        self.image_size = check_img_size(self.image_size, s=self.stride)  # check img_size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names

    def _preprocess_frames(self, frames):
        """
        This method prepare frames for detection. This preprocessing consist three steps:
        1) Stack frames in a proper format for detection
        1) Put the frame in GPU
        2) Normalize image from (0, 255) to (0, 1.0)
        3) Expand or un-squeeze frames
        :param frames:
        :return:
        """

        # Padded resize
        images = [letterbox(frame, self.image_size, stride=self.stride)[0] for frame in frames]
        # images = np.array(images)

        # Stack
        images = np.stack(images, 0)

        # Convert
        images = images[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to 3x416x416
        images = np.ascontiguousarray(images)

        # Put images in GPU
        images = torch.from_numpy(images).to(self.device)
        images = images.half() if self.half else images.float()  # uint8 to fp16/32

        # Normalize images
        images /= 255.0  # 0 - 255 to 0.0 - 1.0
        if images.ndimension() == 3:
            images = images.unsqueeze(0)

        return images

    def _infer(self, images):
        """
        This method detects objects in images
        :param images:
        :return:
        """

        predictions = self.model(images, augment=Cfg.AUGMENT)[0]

        # Apply NMS
        predictions = non_max_suppression(predictions, Cfg.CONFIDENCE_THRESHOLD, Cfg.IOU_THRESHOLD,
                                          Cfg.CLASSES, Cfg.AGNOSTIC_NMS,
                                          max_det=Cfg.MAX_DETECTION)

        return predictions

    @staticmethod
    def _postprocess_predictions(predictions, images, frames):
        """
        This method postprocess predictions
        :param predictions: The output predictions of YOLOv5 model
        :param images: The preprocessed frames
        :param frames: The original frames
        :return:
        """

        detections = []

        # detections per image
        for detection, image, frame in zip(predictions, images, frames):
            if len(detection):
                # Rescale boxes from img_size to im0 size
                detection[:, :4] = scale_coords(image.shape[1:], detection[:, :4], frame.shape).round()
                detection = np.array(detection.cpu())
                detections.append(detection)

        return np.array(detections)

    def detect_objects(self, frames):
        """
        This method detects the objects in batch of frames.
        :param frames:
        :return:
        """

        # Preprocess frames
        images = self._preprocess_frames(frames=frames)

        # Inference
        predictions = self._infer(images=images)

        # Postprocess detections
        # This function has three inputs. The first one is the predictions of the model.
        # The second one is the preprocessed frames (images)
        # The third one is the original frames (frames)
        predictions = self._postprocess_predictions(predictions=predictions, images=images, frames=frames)

        return predictions




def detect():
    source, weights, view_img, save_txt, imgsz = Cfg.SOURCE, Cfg.WEIGHTS, Cfg.VIEW_IMAGE, Cfg.SAVE_TEXT,\
                                                 Cfg.IMAGE_SIZE
    save_img = not Cfg.NO_SAVE and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(Cfg.PROJECT) / Cfg.NAME, exist_ok=Cfg.EXIST_OK)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(Cfg.DEVICE)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=Cfg.AUGMENT)[0]

        # Apply NMS
        pred = non_max_suppression(pred, Cfg.CONFIDENCE_THRESHOLD, Cfg.IOU_THRESHOLD,
                                   Cfg.CLASSES, Cfg.AGNOSTIC_NMS,
                                   max_det=Cfg.MAX_DETECTION)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if Cfg.SAVE_CROP else im0  # for opt.save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if Cfg.SAVE_CONFIDENCE else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or Cfg.SAVE_CROP or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if Cfg.HIDE_LABELS else (names[c] if Cfg.HIDE_CONFIDENCE else
                                                              f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=Cfg.LINE_THICKNESS)
                        if Cfg.SAVE_CROP:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
