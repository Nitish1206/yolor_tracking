import cv2
from track_car import Car_trcker
from utils.datasets import letterbox
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from utils.datasets import LoadStreams, LoadImages
from utils.general import (non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *
import ast
import colorsys




class car_tracker():
    def __init__(self) :
        self.img_size=1280
        self.device='0' #  default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        self.cfg='cfg/yolor_p6.cfg'#, help='*.cfg path')
        self.weights=['yolo_custom.pt']#, help='model.pt path(s)')
        self.names='data/coco.names'#, help='*.cfg path')
        self.classes='classes.txt' #, help='classes path -> [0, 1, 3, 7]')
        self.source=r"N:\Projects\yolor_tracking\videos\test3.mp4"#', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam


    def load_classes(self,path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
    
    def main(self):

       
        device = select_device(self.device)
        
        half = device.type != 'cpu'  # half precision only supported on CUDA
        t0 = time.time()
        
        model = Darknet(self.cfg, self.img_size).cuda()
        model.load_state_dict(torch.load(self.weights[0], map_location=device)['model'])
        #model = attempt_load(weights, map_location=device)  # load FP32 model
        #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        model.to(device).eval()
        if half:
            model.half()

        img = torch.zeros((1, 3, self.img_size, self.img_size), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        names : np.ndarray = np.array(self.load_classes(self.names))
        # names = names[desired_classes]
        desired_classes : list = ast.literal_eval(open( self.classes).read().strip())
        assert isinstance(desired_classes, list), "classes.txt does not contain a list, unsupported type -> " + type(desired_classes)
        colors : np.ndarray = np.array([None] * len(names))
        # Unique colors for different labels
        desired_colors : list = [tuple(np.array(colorsys.hsv_to_rgb(hue,1,1))[::-1]*255) 
                                        for hue in np.arange(0, 1, 1/len(desired_classes))]
        colors[desired_classes] = desired_colors

        cap=cv2.VideoCapture(self.source)


        while True:
            ret, frame = cap.read()

            if not ret :
                break
            img0 = frame.copy()
            im0s=img0.copy()
            img = [letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in img0]

            # Stack
            img = np.stack(img, 0)

            # Convert
            img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1_detect = time_synchronized()
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
                    
                p, s, im0 = "path[i]", '%g: ' % i, im0s[i].copy()
               

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
                    c1=(int(xyxy[0]), int(xyxy[1]))
                    c2=(int(xyxy[2]), int(xyxy[3]))
                    cv2.rectangle(frame, c1, c2, (255, 255, 255), -1, cv2.LINE_AA)      # white filled

                    # print(f"{xyxy = }, {conf = }, {class_id = }")

                    # add bbox coordinates to rect list for centroid tracking
                    # track_rects[class_id].append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                    # add scores to scores list
                    # scores[class_id].append(conf)
                    
                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    # tracker = dlib.correlation_tracker()
                    # rect = dlib.rectangle(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                    # tracker.start_track(frame_rgb, rect)

                    # # add the tracker to our list of trackers so we can
                    # # utilize it during skip frames
                    # trackers_dlib[class_id].append(tracker)
                    # print(f"{trackers_dlib = }")

                    # if save_txt:  # Write to file
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            cv2.imshow(frame,frame)
            if cv2.waitKey(1) & 0xFF==ord("d"):
                break

        cap.release()
        cv2.destroyAllWindows()

car_tracker_obj=Car_trcker()
print(car_tracker_obj)
car_tracker_obj.main()
