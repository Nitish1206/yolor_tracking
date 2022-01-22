""" SAMPLE RUN COMMAND using yolor:
python detect_yolor.py --agnostic-nms --device=0 --source test_video/SRSF-12-OvalWestFootpath_out.mkv 
"""


import argparse
import os
import platform
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from utils.datasets import LoadStreams, LoadImages
from utils.general import (non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

from collections import deque
import dlib
import colorsys
import ast
import traceback

from trackableobject import TrackableObject, TrackableObjects_singleton
from centroidtracker import CentroidTracker
from counter import Counter
from status import TrackStatus, Mode
from draw_frame import draw_frame
from utils.plots import plot_one_box


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
        
    return list(filter(None, names))  # filter removes empty strings (such as last line)

class Car_trcker():
    def __init__(self) -> None:
        self.weights=['yolo_custom.pt']#, help='model.pt path(s)')
        # self.source=r"N:\Projects\yolor_tracking\videos\test3.mp4"#', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
        self.source=r"0"#', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
        self.output="" #', type=str, default='inference/output', help='output folder')  # output folder
        self.img_size=1280 #', type=int, default=1280, help='inference size (pixels)')
        self.conf_thres=0.4 #', type=float, default=0.4, help='object confidence threshold')
        self.iou_thres = 0.5 #', type=float, default=0.5, help='IOU threshold for NMS')
        self.device='0' #  default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        self.view_img = True #', action='store_true', help='display results')
        self.save_txt=False #', action='store_true', help='save results to *.txt')
        self.classes='classes.txt' #, help='classes path -> [0, 1, 3, 7]')
        self.agnostic_nms = True #', action='store_true', help='class-agnostic NMS')
        self.augment=False #', action='store_true', help='augmented inference')
        self.update=False #help='update all models')
        self.cfg='cfg/yolor_p6.cfg'#, help='*.cfg path')
        self.names='data/coco.names'#, help='*.cfg path')
        # Track params
        self.skip_frames=20 #", type=int, default=20, help="# of skip frames between detections")
        self.track_buffer= 30 #", type=int, default=30, help="max buffer size")
        self.track_option=-1 #", type=int, default=-1, help="-1 -> none, 0 -> up/down; 1 -> left/right; 2 -> both")
        self.track_dist=100 #", type=int, default=100, help="Centroid tracking max track distance")
        self.track_frames= 30 #", type=int, default=30, help="Centroid tracking max disappered from frame")
        self._frame_height=480
        self._frame_width=640
        self._tl = (0.002 * (self._frame_height + self._frame_width) / 2)   # line thickness const wrt dimensions
        self._tf = max(self._tl, 1) 

    def detect(self, save_img=False):
    
        # Unpack Detection params
        out, source, weights, view_img, save_txt, imgsz, cfg, names, desired_classes = \
            self.output, self.source, self.weights, self.view_img, self.save_txt, self.img_size, self.cfg, self.names, self.classes
       
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
        
        # Unpack Track params
        skip_frames = self.skip_frames
        TRACK_BUFFER = int(self.track_buffer)
        TRACK_OPTION = TrackStatus(int(self.track_option))
        # Centroid Tracking params
        maxDisappeared, maxDistance = self.track_frames, self.track_dist

        # Initialize device
        device = select_device(self.device)
       
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = Darknet(cfg, imgsz).cuda()
        model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
        #model = attempt_load(weights, map_location=device)  # load FP32 model
        #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        model.to(device).eval()
        if half:
            model.half()  # to FP16

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz, auto_size=64)

        # Get names and colors
        names : np.ndarray = np.array(load_classes(names))
        # names = names[desired_classes]
        desired_classes : list = ast.literal_eval(open(desired_classes).read().strip())
        assert isinstance(desired_classes, list), "classes.txt does not contain a list, unsupported type -> " + type(desired_classes)
        colors : np.ndarray = np.array([None] * len(names))
        # Unique colors for different labels
        desired_colors : list = [tuple(np.array(colorsys.hsv_to_rgb(hue,1,1))[::-1]*255) 
                                        for hue in np.arange(0, 1, 1/len(desired_classes))]
        colors[desired_classes] = desired_colors
        
        # Centroid Tracking objects for each label
        ct_objs = [CentroidTracker(maxDisappeared=maxDisappeared, maxDistance=maxDistance) if i in desired_classes else None for i in range(len(names))]
        
        # A list to store the dlib correlation trackers for each label
        # trackers_dlib = [[] if i in desired_classes else None for i in range(len(names))]
        # A dictionary which maps an objectID to a TrackableObject
        # Keeps id till present in frame
        trackableObjects = TrackableObjects_singleton._trackableObjects

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        
        t_start = time.perf_counter()
        # Run through each frame if dataset.mode == 'video'
        for path, img, im0s, vid_cap in dataset:

            try:
                t1 = time_synchronized()
                
                frame : np.ndarray = np.copy(im0s)
                frame_rgb : np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                vid_cap : cv2.VideoCapture

                if dataset.frame == 1:      # first frame only, initilization
                    frame_height, frame_width =  frame.shape[:2]
                    # Line centre
                    line_centre = [frame_width//2, frame_height//2]
                    # print("line centre==>>", line_centre)
                    # Singleton Thread class for movement counting & computing dwell time
                    # Singleton Thread Class to draw frame

                # initialize the current MODE along with our list of bounding
                # box rectangles returned by either 
                # (1) our object detector or
                # (2) the correlation trackers
                MODE = Mode.waiting

                track_rects = [[] if i in desired_classes else None for i in range(len(names))] # reset after every frame
                scores = [[] if i in desired_classes else None for i in range(len(names))]

                # Refer link for mode change algo
                # https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/
                # detect in 1 of skip_frames or if there are no dlib trackers to track
                if ((dataset.frame - 1) % skip_frames == 0) or \
                    (sum(len(i) if i is not None else 0 for i in trackers_dlib) == 0):
                    MODE = Mode.detecting
                # otherwise, we should utilize our object *trackers* rather than
                # object *detectors* to obtain a higher frame processing throughput
                else:
                    # MODE = Mode.detecting
                    MODE = Mode.tracking
                print(f"{MODE}")
                
                t2 = time_synchronized()
                print(f"{'start':<10}" + ' (%7.3fms)' % ((t2 - t1)*1000))
                
                if MODE is Mode.detecting:
                # Fill track_rects, scores, trackers_dlib 
                    t1 = time_synchronized()
                    # initialize our new set of object trackers
                    trackers_dlib = [[] if i in desired_classes else None for i in range(len(names))]

                    # Perform the actual detection by running the model with the image as input
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    t1_detect = time_synchronized()
                    print(type(img))
                    print(img.shape)
                    exit()
                    pred = model(img, augment=self.augment)[0]
                    # print(f"inference {pred = }")

                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=desired_classes, agnostic=self.agnostic_nms)
                    t2_detect = time_synchronized()
                    # print(f"NMS {pred = }")

                    # Apply Classifier
                    # print(f"final {pred = }")
                    # assert(len(pred) <= 1), "Predictions list has more than 1 elements " + str(len(pred))

                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        det : torch.Tensor
                        # print(f"{i} -> \n {det = } \n {type(det) = }")
                        
                        if webcam:  # batch_size >= 1
                            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                        else:
                            p, s, im0 = path, '', im0s

                        # print(f"{img.shape = } \n{type(img.shape) = }")
                        s += '%gx%g ' % img.shape[2:]  # print string
                        
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        # print(f"{gn = } \n{type(gn) = }")
                        
                        if (det is None) or (len(det) == 0):
                            continue

                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        # print(f"{det = } \n{type(det) = }")

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += '%g %ss, ' % (n, names[int(c)])  # add to string

                        # fill tracking parameters from detection
                        # Write results to txt
                        for *xyxy, conf, class_id in det:
                            
                            class_id = int(class_id)
                            
                            conf = round(float(conf), 2)
                            # print(f"{xyxy = }, {conf = }, {class_id = }")

                            # add bbox coordinates to rect list for centroid tracking
                            track_rects[class_id].append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                            # add scores to scores list
                            scores[class_id].append(conf)
                            
                            # construct a dlib rectangle object from the bounding
                            # box coordinates and then start the dlib correlation
                            # tracker
                            tracker = dlib.correlation_tracker()
                            rect = dlib.rectangle(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                            tracker.start_track(frame_rgb, rect)

                            # add the tracker to our list of trackers so we can
                            # utilize it during skip frames
                            trackers_dlib[class_id].append(tracker)
                            # print(f"{trackers_dlib = }")

                            # if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            # c1=(xywh[0],xywh[1])
                            # c2=(xywh[0]+xywh[2],xywh[1]+xywh[3])
                            # cv2.rectangle(frame, c1, c2, (255, 255, 255), 1, cv2.LINE_AA) 
                                # with open(txt_path + '.txt', 'a') as f:
                                    # f.write(('%g ' * 5 + '\n') % (class_id, *xywh))  # label format

                    t2 = time_synchronized()
                    print('%sDone. (%7.3fs)' % (s, t2_detect - t1_detect))   # Print inference + NMS time
                    print(f"{'detect':<10}" + ' (%7.3fms)' % ((t2 - t1)*1000))

                if MODE is Mode.tracking:
                # Fill track_rects, Use trackers_dlib
                    t1 = time_synchronized()
                    # in *Tracking* MODE we should utilize our object *trackers* rather than
                    # object *detectors* to obtain a higher frame processing throughput
                    for class_id in range(len(names)):
                        if trackers_dlib[class_id] is None:
                            continue
                        
                        for tracker in trackers_dlib[class_id]:
                            # update the tracker and grab the updated position
                            tracker.update(frame_rgb)
                            pos = tracker.get_position()

                            # unpack the position object
                            xmin = int(pos.left())
                            ymin = int(pos.top())
                            xmax = int(pos.right())
                            ymax = int(pos.bottom())

                            # add the bounding box coordinates to the rectangles list
                            track_rects[class_id].append([xmin, ymin, xmax, ymax])
                            # cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255, 255, 255), 1, cv2.LINE_AA) 

                    t2 = time_synchronized()
                    print(f"{'track':<10}" + ' (%7.3fms)' % ((t2 - t1)*1000))

                t1 = time_synchronized()
                
                # use the centroid tracker to associate the 
                # (1) old object centroids with 
                # (2) the newly computed object centroids
                for (class_id, ct) in enumerate(ct_objs):
                    if ct is None:
                        continue

                    # objects -> CentroidTracker().objects 
                    objects = ct.update(track_rects[class_id])
                
                    # loop over the tracked objects
                    for (object_id, centroid) in objects.items():
                        # check to see if a trackable object exists for the current object ID
                        to : TrackableObject = trackableObjects.get((class_id, object_id), None)

                        # if there is no existing trackable object, create one
                        if to is None:
                            to = TrackableObject(object_id, class_id, centroid, TRACK_BUFFER, line_centre)
                        # otherwise, there is a trackable object so we can utilize it
                        # to determine direction
                        else:
                            to.update(centroid)

                        # store the trackable object in our dictionary
                        trackableObjects[(class_id, object_id)] = to

                t2 = time_synchronized()
                print(f"{'cntrd trk':<10}" + ' (%7.3fms)' % ((t2 - t1)*1000))
                t1 = time_synchronized()

                # Loop over the trackable objects dictionary
                for (class_id, object_id) in trackableObjects.copy().keys():
                    to : TrackableObject = trackableObjects[(class_id, object_id)]
                    # if Object id has been deregistered -> delete trackable object
                    if object_id in ct_objs[class_id].deregistered_ids:
                        to.is_deregistered = True

                     # compute the thickness of the line and draw the connecting lines
                    pts : deque = to.centroids
                    for i in np.arange(1, len(pts)):
                        thickness = int(np.sqrt(self.track_buffer / float(i + 1)) * self._tl)
                        cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), colors[class_id], thickness)
                    
                    c1 = pts[0].copy()
                    # draw the ID of the object on the output frame
                    text = "{}_{}".format(class_id, object_id)
                    t_size = cv2.getTextSize(text, 0, fontScale=self._tl/6, thickness=int(self._tf))[0]
                    c1[0] += 5*self._tl
                    c1[1] += int(t_size[1]/2)
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(frame, c1, c2, (255, 255, 255), -1, cv2.LINE_AA)      # white filled
                    cv2.putText(frame, text, (c1[0], c1[1] - 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, self._tl/6, (0, 0, 0), int(self._tf/1.5), lineType=cv2.LINE_AA)                # draw the centroid of the object on the output frame
                    
                    cv2.circle(frame, tuple(pts[0]), int(3*self._tl), colors[class_id] , -1)
                    cv2.circle(frame, tuple(pts[0]), int(2*self._tl), (0, 0, 0) , -1)
                
                # Counter Data from Counter Thread
                if track_rects[class_id]:
                    for i in range(len(track_rects[class_id])):
                        # Get bbox coordinates & label
                        xyxy = track_rects[class_id][i]
                        label = names[class_id]
                        if MODE is Mode.detecting:
                            text_label = '%s: %d%%' % (label, scores[class_id][i] * 100)    # Example: 'person: 72%'
                        elif MODE is Mode.tracking:
                            text_label = '%s' % (label)  # Example: 'person'
                        plot_one_box(xyxy, frame, label=text_label, color=colors[class_id]) # Draw bbox & label
                
                t_end = time.perf_counter()
                frame_rate_calc = 1/(t_end - t_start)
                t_start = time.perf_counter()

                # if save_img or view_img:
                    # draw_frame_obj.update(frame, line_centre, MODE, track_rects, scores, frame_rate_calc)

                if dataset.mode == 'video' and not webcam:
                    save_path = str(Path(out) / Path(path).name)

                # Stream results
                if view_img:
                    print("view image")
                    cv2.imshow(p, frame)
                    if cv2.waitKey(1) & 0xFF==ord("q"):
                        break 

                t2 = time_synchronized()
                print(f"{'draw':<10}" + ' (%7.3fms)' % ((t2 - t1)*1000))
                print(f"FPS -->> {frame_rate_calc}")

                if dataset.frame == dataset.nframes:
                    print("-" * 100, end="\n" * 2)
                    print("Reached end of video")
    
            except KeyboardInterrupt:
                break

            except Exception as e:
                print("EXCEPTION -> ", e)
                traceback.print_exc()
                break

        print("-" * 100, end="\n" * 2)

        if save_txt or save_img:
            print('Results saved to %s' % Path(out))
            
        print('Done. (%.3fs)' % (time.time() - t0))
        print("Cleaning up ...")
        
        print("Stopping Threads ...")
        
        print("Releasing video object instances ...")
        if isinstance(vid_cap, cv2.VideoCapture):
            vid_cap.release()
            print(f" Video Capture {vid_cap} released")
            cv2.destroyAllWindows()
        if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()
            print(f" Video Writer {vid_writer} released")


if __name__ == '__main__':
    

    with torch.no_grad():
        car_tracker_obj= Car_trcker()
        car_tracker_obj.detect()
