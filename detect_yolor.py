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

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def detect(opt, save_img=False):
    
    # Unpack Detection params
    out, source, weights, view_img, save_txt, imgsz, cfg, names, desired_classes = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names, opt.classes
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    
    # Unpack Track params
    skip_frames = opt.skip_frames
    TRACK_BUFFER = int(opt.track_buffer)
    TRACK_OPTION = TrackStatus(int(opt.track_option))
    # Centroid Tracking params
    maxDisappeared, maxDistance = opt.track_frames, opt.track_dist

    # Initialize
    device = select_device(opt.device)
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Darknet(cfg, imgsz).cuda()
    model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
    #model = attempt_load(weights, map_location=device)  # load FP32 model
    #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

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
    trackers_dlib = [[] if i in desired_classes else None for i in range(len(names))]
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
                counter_obj = Counter(line_centre, TRACK_OPTION, names)
                counter_obj.start()
                # Singleton Thread Class to draw frame
                draw_frame_obj : draw_frame = draw_frame(TRACK_OPTION, frame_height, frame_width, 
                                                         names, colors, TRACK_BUFFER).start()

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
                pred = model(img, augment=opt.augment)[0]
                # print(f"inference {pred = }")

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=desired_classes, agnostic=opt.agnostic_nms)
                t2_detect = time_synchronized()
                # print(f"NMS {pred = }")

                # Apply Classifier
                if classify:
                    pred = apply_classifier(pred, modelc, img, im0s)
                    # print(f"classify {pred = }")

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

                    txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                    
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

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (class_id, *xywh))  # label format

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
            
            # Counter Data from Counter Thread
            counter_text = str(dict(filter(lambda x: x[1] != 0 and x[0] in names[desired_classes], counter_obj.counter.items())))
            dwell_text = str(dict(filter(lambda x: not not x[1] and x[0] in names[desired_classes], counter_obj.dwell_time.items())))
            (left_text, right_text, up_text, down_text) = None, None, None, None
            if TRACK_OPTION != TrackStatus.none:
                if TRACK_OPTION is TrackStatus.left_right or TRACK_OPTION is TrackStatus.both: 
                    left_text = f"{str(counter_obj.left_movement.get_object_ids())}"
                    right_text = f"{str(counter_obj.right_movement.get_object_ids())}"
                if TRACK_OPTION is TrackStatus.up_down or TRACK_OPTION is TrackStatus.both: 
                    up_text = f"{str(counter_obj.up_movement.get_object_ids())}"
                    down_text = f"{str(counter_obj.down_movement.get_object_ids())}"
            
            t_end = time.perf_counter()
            frame_rate_calc = 1/(t_end - t_start)
            t_start = time.perf_counter()

            if save_img or view_img:
                draw_frame_obj.update(frame, line_centre, MODE, track_rects, scores, frame_rate_calc, 
                                      (counter_text, dwell_text, left_text, right_text, up_text, down_text))

            if dataset.mode == 'video' and not webcam:
                save_path = str(Path(out) / Path(path).name)

            # Stream results
            if view_img:
                input_key = cv2.waitKey(1) & 0xFF
                cv2.imshow(p, frame)
                # centre line movement
                if TRACK_OPTION != TrackStatus.none:
                    if TRACK_OPTION is TrackStatus.left_right or TRACK_OPTION is TrackStatus.both: 
                        if input_key == ord('a'):   # left
                            line_centre[0] -= 5
                        elif input_key == ord('d'): # right
                            line_centre[0] += 5
                    if TRACK_OPTION is TrackStatus.up_down or TRACK_OPTION is TrackStatus.both: 
                        if input_key == ord('w'): # up
                            line_centre[1] -= 5
                        elif input_key == ord('s'): # down
                            line_centre[1] += 5
                    if input_key == ord('r'): # reset centre line
                        line_centre = [frame_width//2, frame_height//2]
                    Counter.reference_line = line_centre
                # Exit
                elif input_key == ord('q') or input_key == 27:
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        
                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(frame)

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
        # if not opt.update:  
        #     os_ = platform.system().lower()
        #     if os_ == 'darwin':             # MacOS
        #         os.system('open ' + save_path)
        #     elif os_ == "windows":          # Windows
        #         os.system("explorer.exe " + str(Path(out)))
        #         os.system(save_path)
        #     elif os_ == "linux":
        #         os.system("nautilus " + str(Path(out)))

    print('Done. (%.3fs)' % (time.time() - t0))
    print("Cleaning up ...")

    print("Creating Output files ...")
    with open((os.path.splitext(save_path)[0] + ".txt"), mode="w+", encoding = "utf-8") as f:
        f.write(Path(save_path).name + "\n\nCounter: \n" + counter_text + "\n\nDwell time: \n" + dwell_text + "\n")
    
    print("Stopping Threads ...")
    if isinstance(counter_obj, Counter):
        counter_obj.stop_thread()
    if isinstance(draw_frame_obj, draw_frame):
        draw_frame_obj.stop_thread()
    
    print("Releasing video object instances ...")
    if isinstance(vid_cap, cv2.VideoCapture):
        vid_cap.release()
        print(f" Video Capture {vid_cap} released")
    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()
        print(f" Video Writer {vid_writer} released")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Detect params
    parser.add_argument('--weights', nargs='+', type=str, default=['yolo_custom.pt'], help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', type=str, default='classes.txt', help='classes path -> [0, 1, 3, 7]')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    # Track params
    parser.add_argument("--skip_frames", type=int, default=20, help="# of skip frames between detections")
    parser.add_argument("--track_buffer", type=int, default=30, help="max buffer size")
    parser.add_argument("--track_option", type=int, default=-1, help="-1 -> none, 0 -> up/down; 1 -> left/right; 2 -> both")
    parser.add_argument("--track_dist", type=int, default=100, help="Centroid tracking max track distance")
    parser.add_argument("--track_frames", type=int, default=30, help="Centroid tracking max disappered from frame")
    
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect(opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt)
