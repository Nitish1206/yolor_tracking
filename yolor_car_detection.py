import cv2
from cv2 import cvtColor
from grpc import server
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
from numba import cuda
from queue import Queue
from car_utils.get_parking_area import parking_rects
from car_utils.helper import *
from car_utils.parkings import Parking
from centroidtracker import CentroidTracker
import dlib

cuda.select_device(0)
cuda.close()
cuda.select_device(0)


class car_tracker():

    def __init__(self):
        self.img_size=1280
        self.auto_size=64
        self.device_cfg='0' #  default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        self.cfg='cfg/yolor_p6.cfg'#, help='*.cfg path')
        self.weights=['yolo_custom.pt']#, help='model.pt path(s)')
        self.names_path='data/coco.names'#, help='*.cfg path')
        self.classes='classes.txt' #, help='classes path -> [0, 1, 3, 7]')

        self.augment=False #', 
        self.conf_thres=0.4 #', type=float, default=0.4, help='object confidence threshold')
        self.iou_thres = 0.5 #', type=float, default=0.5, help='IOU threshold for NMS')
        self.agnostic_nms = True #', action='store_true', help='class-agnostic NMS')

        self.device = select_device(self.device_cfg)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = Darknet(self.cfg, self.img_size).cuda()
        self.model.load_state_dict(torch.load(self.weights[0], map_location=self.device)['model'])
        
        self.model.to(self.device).eval()
        if self.half:
            self.model.half()

        img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
        self.names : np.ndarray = np.array(self.load_classes(self.names_path))
        self.desired_classes : list = ast.literal_eval(open( self.classes).read().strip())
        assert isinstance(self.desired_classes, list), "classes.txt does not contain a list, unsupported type -> " + type(self.desired_classes)
        self.colors : np.ndarray = np.array([None] * len(self.names))
        
        desired_colors : list = [tuple(np.array(colorsys.hsv_to_rgb(hue,1,1))[::-1]*255) 
                                        for hue in np.arange(0, 1, 1/len(self.desired_classes))]
        self.colors[self.desired_classes] = desired_colors
        self.ct = CentroidTracker()

        self.set_values_from_server()
       
    
    def set_values_from_server(self,source=r"N:\Projects\yolor_tracking\videos\test3.mp4",parking_dict={},server_flag=False):

        self.source= source #', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam

        self.frame_queue = Queue(maxsize=1)      
        self.current_car_rects=[]
        self.current_centroid=[]
        self.parkng_rect_obj=parking_rects()
        self.processing_flag=True
        self.processing_thread = Thread(target=self.process_frame)
        self.processing_thread.start()
        self.parking_rects=[]
        self.server=False
        self.parking_objects={}
        self.parking_dicts={}
        self.tracking_rects=[]
        self.dlibtrackers={}
        self.create_parking_objects(parking_dict,server_flag)
        self.set_parking_variable()
        self.mainprocess_flag=True
        self.progress=0

    def set_parking_variable(self):
        self.checkout_buffer=10  #secs
        self.park_buffer_threshold=5
        self.park_raw_buffer_threshold=5
    

    def load_classes(self,path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  # filter removes empty strings (such as last line)
    
    def create_parking_objects(self,parking_dict,server_flag):
        if not server_flag:
            if len(self.parking_dicts.keys()) == 0:
                self.parkng_rect_obj.get_coordinate()
                park_rect,frame = self.parkng_rect_obj.return_area()
                parking_dict= {int(i//4)+1: park_rect[i:i+4] for i in range(0, len(park_rect),4)}
                    
        for park_ids in parking_dict.keys():
            parking_obj=Parking()
            parking_obj.id=park_ids
            rects=self.parking_dicts[park_ids]
            parking_obj.rects=rects
            parking_obj.xs=[x[0] for x in rects]
            parking_obj.ys=[x[1] for x in rects]
            parking_centre=(sum(parking_obj.xs)//4,sum(parking_obj.ys)//4)
            parking_obj.centroid=parking_centre
            self.parking_objects[park_ids]=parking_obj


    def main(self):
        
        cap=cv2.VideoCapture(self.source)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) % 100
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _, self.imgs_ = cap.read()

        while self.mainprocess_flag:

            ret, frame = cap.read()

            if not ret :
                break
            
            frameId = int(round(cap.get(1)))
            self.progress=(frameId/length)*100
            current_frame_time = frameId/fps
            rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            if self.frame_queue.full():
                self.frame_queue.get()

            self.frame_queue.put(frame)
               
            
            for key in self.parking_objects.keys():
                
                updated_rects=[]

                parkings=self.parking_objects[key]
                print("parkings---",parkings)
                rect=parkings.rects
                xs=parkings.xs
                ys=parkings.ys
                print("key",parkings.parking_status)
                # parking_centre=(sum(xs)//4,sum(ys)//4)

                if not parkings.parking_status:
                    cv2.circle(frame, tuple(parkings.centroid), 3, (0,0,255) , -1)
                    pts = np.array([parkings.rects],np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    isClosed = True
                    # Blue color in BGR
                    color = (255, 0, 0)
                    thickness = 2
                    cv2.polylines(frame, [pts], isClosed, color, thickness)  
                    
                    for k,centr in enumerate(self.current_centroid):
                        cv2.circle(frame, tuple(centr), 3, (255,0,255) , -1)

                        park_status,coords = if_is_inside(xs,ys,centr)
                        if park_status:
                            current_car_rect=self.current_car_rects[k]
                            print("current car rect",len(current_car_rect))
                            # parkings.rects=current_car_rect
                            # self.tracking_rects.append(current_car_rect)
                            t = dlib.correlation_tracker()
                            dlib_rect = dlib.rectangle(current_car_rect[0], current_car_rect[1], current_car_rect[2], current_car_rect[3])
                            t.start_track(rgb_frame, dlib_rect)
                            self.dlibtrackers[parkings.id]=t
                            parkings.parking_status=True
                            parkings.occupied_center=centr
                         
                elif parkings.parking_status:
                    
                    # parkings.park_in_buffer=0
                    t=self.dlibtrackers[parkings.id]
                    t.update(rgb_frame)
                    pos = t.get_position()
                    # unpack the position object
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())
                    updated_rects.append([startX,startY,endX,endY])
                    centroid=((startX+endX)//2,endY)
                    # cv2.circle(frame, tuple(centroid), 3, (0,255,25) , -1)

                    park_status,coords = if_is_inside(xs,ys,centroid)
                    if park_status and not parkings.park_in_status:
                        parkings.park_in_buffer+=1
                        if parkings.park_in_buffer > self.park_buffer_threshold:
                            parkings.park_in_status=True
                            parkings.park_in_buffer=0
                            # cv2.circle(frame, tuple(parkings.centroid), 3, (0,255,25) , -1)

                    if not park_status:
                        parkings.park_out_buffer+=1
                        if parkings.park_out_buffer>self.park_buffer_threshold:
                            parkings.parking_status=False
                            parkings.park_out_buffer=0
                            del self.dlibtrackers[parkings.id]
                
                if parkings.park_in_status:

                    cv2.circle(frame, tuple(parkings.centroid), 3, (0,255,25) , -1)

                if not parkings.parking_status:
                    cv2.circle(frame, tuple(parkings.centroid), 3, (0,25,255) , -1)

            cv2.imshow("frame",frame)
            if cv2.waitKey(1) & 0xFF==ord("q"):
                break
        
        self.processing_flag=False
        
        cap.release()
        cv2.destroyAllWindows()

    def stop_main_process(self):
        self.mainprocess_flag=False

    
    @torch.no_grad()
    def detect_car_rect(self,frame):

        img0 = frame.copy()
        im0s=img0.copy()
        
        img = letterbox(img0, new_shape=self.img_size,auto_size=self.auto_size)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416

        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3: 
            img = img.unsqueeze(0)

        pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.desired_classes, agnostic=self.agnostic_nms)
        self.current_car_rects=[]
        self.current_centroid=[]
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            det : torch.Tensor
            im0 = im0s.copy()
            
            if (det is None) or (len(det) == 0):
                continue

            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, class_id in det:
                
                class_id = int(class_id)
                conf = round(float(conf), 2)
                r1=[int(xyxy[0]), int(xyxy[1]),int(xyxy[2]), int(xyxy[3])]
                
                c1=(int(xyxy[0])+int(xyxy[2]))//2
                c2=int(xyxy[3])
                self.current_car_rects.append(r1)
                self.current_centroid.append([c1,c2])

    def process_frame(self):
        while self.processing_flag:
            if not self.frame_queue.empty():
                small_frame = self.frame_queue.get()
                self.detect_car_rect(small_frame)


# with torch.no_grad():
if __name__ == '__main__' :
    car_tracker_obj=car_tracker()
    print(car_tracker_obj)
    car_tracker_obj.set_values_from_server()
    car_tracker_obj.main()

    torch.cuda.empty_cache()
