import traceback
from xmlrpc.client import DateTime
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
from car_utils.get_license_number import get_result_api
from centroidtracker import CentroidTracker
import dlib
import csv
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QImage, QPixmap,QColor
from PyQt5.QtCore import pyqtSlot, Qt
import datetime
from constants import *

class car_tracker(QThread):

    # image_signal_ = pyqtSignal(QImage)
    progress_signal = pyqtSignal(int)
    parking_status_signal=pyqtSignal(dict)


    def __init__(self,model_obj):
        super(car_tracker, self).__init__()
        self.set_parking_variable()
        self.ai_=model_obj
        self.car_object_status={}
        
    def set_values_from_server(self,source=r"N:\Projects\yolor_tracking\videos\test3.mp4",parking_dict={},car_data={},pui=None,server_flag=False):

        self.reset_server_parameters(parking_dict,car_data)
        self.processing_thread = Thread(target=self.process_frame)
        self.processing_thread.start()
        self.ui=pui
        # self.create_parking_objects(parking_dict,server_flag)
        self.source= source #', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam

    def reset_server_parameters(self,parking_obj,car_obj):
        self.frame_queue = Queue(maxsize=1)
        self.current_car_rects = []
        self.current_centroid = []
        self.parkng_rect_obj = parking_rects()
        self.processing_flag = True
        self.parking_rects = []
        self.parking_objects = parking_obj
        self.car_objects = car_obj
        self.parking_dicts = {}
        self.tracking_rects = []
        self.dlibtrackers = {}
        self.mainprocess_flag = True
        self.progress = 0
        self.data_to_send = {}
        self.car_rect_centr=[]
        self.car_id_start=0
        self.ut=-1
        self.ct = CentroidTracker()
        

    def set_parking_variable(self):
        self.checkout_buffer=10  #secs
        self.park_buffer_threshold=60
        self.park_buffer_threshold_out=120
        self.park_raw_buffer_threshold=5

    def load_classes(self,path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  # filter removes empty strings (such as last line)
    

    def run(self):
        
        cap=cv2.VideoCapture(self.source)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) % 100
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _, self.imgs_ = cap.read()
        
        video_height = 720
        video_width = 1080
        self.print_strings=""
        # counter=0
        consts=0 if self.ui.comboBox.currentText().split(" ")[1] == "AM" else 12
        print(consts)
        print("N"*50)
        time.sleep(2)
        user_time=(consts+int(self.ui.comboBox.currentText().split(" ")[0]))*3600
        self.user_day = self.ui.comboBox_2.currentText()

        while self.mainprocess_flag:

            ret, frame = cap.read()

            if not ret :
                break

            frame=cv2.resize(frame,(video_width,video_height))
            frameId = int(round(cap.get(1)))
            self.progress = (frameId/length)*100
            self.current_frame_time_unit = user_time + int(frameId/fps)
            conversion = datetime.timedelta(seconds=self.current_frame_time_unit)
            conversion=str(conversion)
            if "," in conversion:
                conversion=conversion.split(",")[-1].strip()
            d = datetime.datetime.strptime(str(conversion), "%H:%M:%S")
            self.current_frame_time = d.strftime("%I:%M:%S %p") 
            self.str_current_frame_time = str(d.strftime("%I:%M:%S %p"))
            # str_current_frame= str(conversion)+" "+meredian
            rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            if self.frame_queue.full():
                self.frame_queue.get()

            self.frame_queue.put(frame)
               
            for key in self.parking_objects.keys():
                
                updated_rects=[]
                parkings=self.parking_objects[key]
                xs=parkings.xs
                ys=parkings.ys
               
                pts = np.array([parkings.rects],np.int32)
                pts = pts.reshape((-1, 1, 2))
                if DRAW_STATUS:
                    isClosed = True
                    # Blue color in BGR
                    color = (255, 0, 0)
                    thickness = 2
                    cv2.polylines(frame, [pts], isClosed, color, thickness)  
                    for rectcentr in self.car_rect_centr:
                        centr=rectcentr[0]
                        cv2.circle(frame, tuple(centr), 3, (255,0,255) , -1)

                if not parkings.parking_status:
                    if DRAW_STATUS:
                        cv2.circle(frame, tuple(parkings.centroid), 3, (0,0,255) , -1)
                    for rectcentr in self.car_rect_centr:
                        try:
                            park_status,coords = if_is_inside(xs,ys,rectcentr[0],parkings.rects)
                        except:
                            traceback.print_exc()
                            print("co ordinate error")
                            self.processing_thread = False
                            self.mainprocess_flag = False
                            break

                        if park_status:
                            
                            self.car_id_start+=1
                            current_car_rect = rectcentr[1]
                            t = dlib.correlation_tracker()
                            dlib_rect = dlib.rectangle(current_car_rect[0], current_car_rect[1], current_car_rect[2], current_car_rect[3])
                            t.start_track(rgb_frame, dlib_rect)
                            self.dlibtrackers[parkings.id] = t
                            parkings.parking_status = True
                            break
                         
                elif parkings.parking_status:
                    
                    parkings.park_in_buffer=0
                    t=self.dlibtrackers[parkings.id]
                    t.update(rgb_frame)
                    pos = t.get_position()
                    # unpack the position object
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())
                    
                    centroid=((startX+endX)//2,endY)

                    park_status,coords = if_is_inside(xs,ys,centroid,parkings.rects)
                    if park_status and not parkings.park_in_status:
                        parkings.park_in_buffer_+=1
                        if parkings.park_in_buffer_ > self.park_buffer_threshold:
                            try:
                                current_car=self.car_objects[parkings.id]
                                # car_image = frame[startY:endY, startX:endX]
                                # number_plate = get_result_api(car_image)

                                parkings.car_id=current_car.id
                                car_number=current_car.number

                                parkings.car_number.append(car_number)
                                parkings.park_in_status = True
                                parkings.park_in_buffer = 0
                                parkings.park_in_time.append(self.current_frame_time_unit)
                                # self.print_strings+="parking id "+str(parkings.id)+" is occupied by car "+ str(car_number) +" at " + str(current_frame_time) + " seconds"+"\n"  
                                ptype=parkings.parking_type
                                current_car.current_parking_type=ptype
                                
                                event_str="car license plate "+str(car_number) +" entered parking "+str(parkings.id) + "\n"+ "parking is "+str(ptype) +"\n"

                                if ptype =="Red Zone":
                                    event_str+=self.check_condition_for_rny_parking(parking_type=ptype,lp=car_number)
                                elif ptype == "Yellow Zone":
                                    event_str+=self.check_condition_for_rny_parking(parking_type=ptype,lp=car_number)
                                elif ptype =="Residential":
                                    event_str+=self.check_conditions_for_residential_parking(parking_type=ptype,permit_type=str(current_car.permit_type),lp=car_number)
                                elif ptype == "villa":
                                    event_str+=self.check_condition_for_villa_parking(parking_type=ptype,permit_type=str(current_car.permit_type),lp=car_number)
                                elif ptype =="Disabled":
                                    event_str += self.check_condition_for_disabled_parking(parking_type=ptype,permit_type=str(current_car.permit_type),lp=car_number)
                                elif ptype == "Premium":
                                    event_str += self.check_condition_for_standard_premium(car_obj=current_car)
                                elif ptype == "Standard":
                                    event_str += self.check_condition_for_standard_premium(car_obj=current_car)
                                
                                data_to_emit={parkings.id:{"event":event_str,"ts_in": self.str_current_frame_time,"ts_out":""},"thread id":str(self.currentThreadId())}
                                self.parking_status_signal.emit(data_to_emit)

                            except:
                                traceback.print_exc()
                                pass

                    elif not park_status:
                        parkings.park_out_buffer+=1
                        if parkings.park_out_buffer > self.park_buffer_threshold_out:
                            try:
                                parkings.parking_status = False
                                parkings.park_in_status = False
                                parkings.park_out_buffer = 0
                                parkings.park_out_time.append(self.current_frame_time_unit)
                                if parkings.car_id !=None:
                                    car=self.car_objects[parkings.car_id]
                                    car_number = car.number
                                    car.previous_parking_type=car.current_parking_type
                                    # self.print_strings+="parking id "+str(parkings.id)+" car "+str(parkings.car_number[-1]) +" left the parking " + str(current_frame_time) +" seconds"+ "\n"  
                                    data_to_emit=data_to_emit = {parkings.id:{"event":"car license plate "+str(car_number) +" left","ts_in":"",
                                    "ts_out": self.str_current_frame_time}}
                                    self.parking_status_signal.emit(data_to_emit)
                                    del self.dlibtrackers[parkings.id]
                            except:
                                traceback.print_exc()
                                pass 
                if DRAW_STATUS:
                    if parkings.parking_status and not parkings.park_in_status:
                        cv2.circle(frame, tuple(parkings.centroid), 3, (255,55,125) , -1)
                    elif parkings.park_in_status:
                        cv2.circle(frame, tuple(parkings.centroid), 3, (0,255,0) , -1)
                    elif not parkings.parking_status:
                        cv2.circle(frame, tuple(parkings.centroid), 3, (0,25,255) , -1)
            
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format.Format_BGR888)
            self.updateFrame(image)
            # self.image_signal.emit(image)
            self.progress_signal.emit(self.progress)
           
        
        self.processing_flag=False
        cap.release()
        cv2.destroyAllWindows()
        # self.process_result()

    def updateFrame(self, image):
        qpix_img  = QPixmap.fromImage(image)
        # set a scaled pixmap to a w x h window keeping its aspect ratio 
        self.ui.video_processor.setPixmap(qpix_img.scaled(self.ui.video_processor.size(),
                                            Qt.AspectRatioMode.KeepAspectRatio))

    def stop_main_process(self):
        self.mainprocess_flag=False

    def process_result(self):
        self.data_to_send["message"]={}
        for key_ in self.parking_objects.keys():
            parkings_=self.parking_objects[key_]
            self.data_to_send["message"][parkings_.id]={}
            self.data_to_send["message"][parkings_.id]["parking_pos"]=parkings_.rects
            self.data_to_send["message"][parkings_.id]["id"]=parkings_.id
            self.data_to_send["message"][parkings_.id]["Number_plate"]=""
            park_in_out=[[int(in_),int(out_)] for in_,out_ in zip(parkings_.park_in_time,parkings_.park_out_time)]
            self.data_to_send["message"][parkings_.id]["park_in_out_time"]=park_in_out
       
        self.write_to_csv()
        print("*"*100)
        print(self.data_to_send)
        print("*"*100)

    def write_to_csv(self):
        
        file1 = open("output/test3.txt","w+")
        file1.writelines(self.print_strings)

    def check_conditions_for_residential_parking(self,parking_type,permit_type,lp):
        result_str=""
        print("pp",parking_type,permit_type)
        if parking_type == permit_type:
            # print("got same parking type")
            result_str = "User has residential permit" + "\n" + "User is not Fined"
        
        elif parking_type != permit_type:
            # print("different permit")
            if self.user_day == "offday":
                result_str = "User doesn't have residential permit" + "\n" + "Today is off day" + "\n" + "User is Fined"
            else:
                # print("N"*50)
                # print(RESIDENTIAL_PARKING_NO_PERMIT_NO_FINE_TIME_START)
                # print(self.current_frame_time_unit)
                # print(RESIDENTIAL_PARKING_NO_PERMIT_NO_FINE_TIME_END)
                # print("D"*50)
                # print(RESIDENTIAL_PARKING_NO_PERMIT_FINE_TIME_START)
                # print(self.current_frame_time_unit)
                # print(RESIDENTIAL_PARKING_NO_PERMIT_FINE_TIME_END)
                # print("N"*50)
                # time.sleep(5)
                # if  RESIDENTIAL_PARKING_NO_PERMIT_NO_FINE_TIME_START_dt < self.current_frame_time > RESIDENTIAL_PARKING_NO_PERMIT_NO_FINE_TIME_END_dt:
                #     result_str = "User doesn't have residential permit "+"\n"+"Today is workday" + "\n" + "time is after 8 a.m and before 9 p.m" +"\n" +"User is not Fined"
                # elif RESIDENTIAL_PARKING_NO_PERMIT_FINE_TIME_START_dt<self.str_current_frame_time>RESIDENTIAL_PARKING_NO_PERMIT_FINE_TIME_END_dt:
                #     result_str = "User doesn't have residential permit "+"\n"+ "Today is workday" + "\n" + "time is after 9 p.m and before 8 a.m" +"\n" +"User is Fined"
                if  RESIDENTIAL_PARKING_NO_PERMIT_NO_FINE_TIME_START < self.current_frame_time_unit < RESIDENTIAL_PARKING_NO_PERMIT_NO_FINE_TIME_END:
                    result_str = "User doesn't have residential permit "+"\n"+"Today is workday" + "\n" + "time is after 8 a.m and before 9 p.m" +"\n" +"User is not Fined"
                elif RESIDENTIAL_PARKING_NO_PERMIT_FINE_TIME_START < self.current_frame_time_unit > RESIDENTIAL_PARKING_NO_PERMIT_FINE_TIME_END:
                    result_str = "User doesn't have residential permit "+"\n"+ "Today is workday" + "\n" + "time is after 9 p.m and before 8 a.m" +"\n" +"User is Fined"
                    
        return result_str

    def check_condition_for_villa_parking(self,parking_type,permit_type,lp):
        result_str=""
        if parking_type == permit_type:
            result_str= "User has villa permit" + "\n" + "User is not Fined"
        elif parking_type != permit_type:
            result_str = "User does not have villa permit" + "\n" + "User is Fined"
        return result_str

    def check_condition_for_disabled_parking(self,parking_type,permit_type,lp):
        result_str=""
        if parking_type == permit_type:
            result_str="User has Disabled permit" + "\n" + "User is not Fined"
        elif parking_type != permit_type:
            result_str ="User does not have Disabled permit" + "\n" + "User is Fined"
        return result_str
    
    def check_condition_for_rny_parking(self,parking_type,lp):
        result_str="User has parked in " + parking_type +"\n"+ "User is Fined"
        return result_str

    def check_condition_for_standard_premium(self,car_obj):
#        12 a.m – 8 a.m.
        result_str=""
        if self.user_day =="offday":
            result_str= "Do nothing"
        else:
            if PREMIMUM_PARKING_NO_PERMIT_NO_FINE_TIME_START < self.current_frame_time_unit < PREMIMUM_PARKING_NO_PERMIT_NO_FINE_TIME_END:
                result_str="time is after 12AM and before 8AM"
            elif PREMIMUM_PARKING_NO_PERMIT_FINE_TIME_START < self.current_frame_time_unit:
                previousparkingtype=car_obj.previous_parking_type
                if previousparkingtype == "Premium" or previousparkingtype == "Standard":
                    result_str="time is after 8AM" 
                else:
                    result_str="time is after 8AM"

        return result_str
        
    @torch.no_grad()
    def detect_car_rect(self,frame):

        img0 = frame.copy()
        im0s=img0.copy()
        
        img = letterbox(img0, new_shape=self.ai_.img_size,auto_size=self.ai_.auto_size)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416

        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.ai_.device)
        img = img.half() if self.ai_.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3: 
            img = img.unsqueeze(0)

        pred = self.ai_.model(img, augment=self.ai_.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.ai_.conf_thres, self.ai_.iou_thres, classes=self.ai_.desired_classes, agnostic=self.ai_.agnostic_nms)
       
        self.car_rect_centr=[]
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
                if TOP_CAMERA:
                    c2=(int(xyxy[1])+int(xyxy[3]))//2
                    centr=[c1,c2]
                    self.car_rect_centr.append([centr,r1])
                if FRONT_CAMERA:
                    c2=int(xyxy[3])
                    centr=[c1,c2]
                    self.car_rect_centr.append([centr,r1])
               
    def process_frame(self):
        while self.processing_flag:
            if not self.frame_queue.empty():
                small_frame = self.frame_queue.get()
                self.detect_car_rect(small_frame)

    def stop_thread(self):
        self.processing_flag=False
        self.stop_main_process()


if __name__ == '__main__' :
    cuda.select_device(0)
    
    video_dir=r"N:\Projects\yolor_tracking\videos"
    # video_name="ion_alarm.mkv"
    video_name="test3.mp4"
    # video_name="Hikvision.mp4"
    # torch.cuda.empty()
    video_path=os.path.join(video_dir,video_name)
    with torch.no_grad():

        car_tracker_obj=car_tracker()
        print(car_tracker_obj)
        car_tracker_obj.set_values_from_server(source=video_path)
        car_tracker_obj.main()

        torch.cuda.empty_cache()
