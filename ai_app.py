from application.windows import roi_window,processing_window
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap,QColor
from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog,QTableWidgetItem,QComboBox,QSizePolicy
from PyQt5 import QtWidgets
import sys
import cv2
import os
from car_utils.parkings import Parking
from car_utils.cars import Car
from random import randint
import torch
from yolor_car_detection_qthread import car_tracker
from glob import glob
import traceback
import json 
from utils.torch_utils import select_device, load_classifier, time_synchronized
from load_model import LoadModel
from constants import *
from application.windows.design import style

cwd=os.getcwd()

def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

class Window(QMainWindow, roi_window.Ui_MainWindow,processing_window.Ui_Form):
    def __init__(self,screenw,screenh, parent=None):
        super(Window, self).__init__()
        self.setupUi(self)
        self.uiscreenWidth=screenw
        self.uiscreenHeight=screenh
        self.setFixedWidth(screenw)
        self.setFixedHeight(screenh)
        # self.setWindowFlags(Qt.WindowStaysOnTopHint)
        
        self.frame_size=(1080,720)
        self.all_parkings={"parking id":[],"parking cordindate":[],"parking type":[]}
        self.parking_data={}
        self.car_data={}
        self.parking_co_ordinates=[]
        self.config_path = CONFIG_PATH
        self.select_config_files()
        self.upload_button.clicked.connect(self.upload_image_)
        self.clear_button.clicked.connect(self.remove_last_cordinate)
        self.save_button.clicked.connect(self.update_parking_data)
        self.save_parking.clicked.connect(self.save_parkings)
        self.frame_.mousePressEvent = self.getPixel
        self.saveandstartButton.clicked.connect(self.open_processing_window)
        self.SaveconfigButton.clicked.connect(self.save_configuration)
        self.select_config_combobox.currentTextChanged.connect(self.updateConfig)
        self.daySelectorCombobox.addItems(DAY_LIST)
        self.startTimeCombobox.addItems(STARTTIMES)
        self.hourCombobox.addItems(HOURDATA)
        self.minutesCombobox.addItems(MINUTESDATA)

        with torch.no_grad():
            self.model_obj=LoadModel()
    
    
    def updateConfig(self, value):
        # print("combobox changed", value)
        if value != "None" or value != "" or value !=" ":
            filePath=self.config_path+os.sep+value
            try:
                with open(filePath) as json_file:
                    data = json.load(json_file)

                self.image_path_=data["image_path"]
                self.parking_co_ordinates=[]
                coordinates=data["parking_properties"]["parking cordindate"]
                for coord in coordinates:
                    for co in coord:
                        self.parking_co_ordinates.append(co)

                self.parkingType=data["parking_properties"]["parking type"]
                day=data["day"]
                starttime=data["starttime"]
                hour=data["hour"]
                minute=data["minute"]
                # self.upload_image_()
                # self.setImageToLabel(self.image_path_)
                self.update_image()
                
                self.update_parking_data(settext=True,ptypelist=self.parkingType)
                self.daySelectorCombobox.setCurrentText(day)
                self.startTimeCombobox.setCurrentText(starttime)
                self.hourCombobox.setCurrentText(hour)
                self.minutesCombobox.setCurrentText(minute)
                self.save_parkings()
            except:
                pass
        # print(data)

    def select_config_files(self):
        config_files=glob(self.config_path+"/*")
        file_names=["None"]

        for files in config_files:
            file_names.append(os.path.basename(files))
        
        self.select_config_combobox.clear()
        self.select_config_combobox.addItems(file_names)
   
    def save_configuration(self):
        try:
            current_config = {}
            current_config["image_path"]=self.image_path_
            all_parkings_={"parking id":[],"parking cordindate":[],"parking type":[]}
            pptype=""
            for parking in self.parking_data.values():
                all_parkings_["parking id"].append(parking.id)
                all_parkings_["parking cordindate"].append(parking.rects)
                all_parkings_["parking type"].append(parking.parking_type)
                pptype+=str(parking.parking_type[0])

            current_config["parking_properties"]=all_parkings_
            day_=self.daySelectorCombobox.currentText()
            starttime_=self.startTimeCombobox.currentText()
            hour_=self.hourCombobox.currentText()
            minute_=self.minutesCombobox.currentText()
            current_config["day"]=day_
            current_config["starttime"]=starttime_
            current_config["hour"]=hour_
            current_config["minute"]=minute_
            # print(pptype , type(pptype))
            # print(day_ , type(day_))
            # print(starttime_ , type(starttime_))
            configName = os.path.basename(self.image_path_).split(".")[0]+"_"+str(len(all_parkings_["parking id"]))+"_"+pptype+"_"+day_+"_"+starttime_
            save_path=self.config_path+os.sep+ configName
            
            with open(save_path+".json", "w") as outfile:
                json.dump(current_config, outfile)
            self.select_config_files()
        except:
            traceback.print_exc()
            pass
        

    def upload_image_(self):
        image_path = QFileDialog.getOpenFileName(
            None, 'Test Dialog', os.getcwd(), 'All Files(*.jpg*)')
        self.image_path_=r""+image_path[0]
        self.setImageToLabel(self.image_path_)
    
    def setImageToLabel(self,image_path):
        try:
            self.frame=cv2.imread(image_path)
            self.frame=cv2.resize(self.frame,self.frame_size)
            self.rgb_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.cvt2qt = QImage(self.rgb_image.data, self.rgb_image.shape[1], self.rgb_image.shape[0], QImage.Format_RGB888)
            self.frame_.setPixmap(QPixmap.fromImage(self.cvt2qt))
        except:
            traceback.print_exc()
            pass

    def getPixel(self, event):
        x = event.pos().x()
        y = event.pos().y()
        label_size=self.frame_.size()
        label_width=label_size.width()
        label_height=label_size.height()
        frame_width=self.frame_size[0]
        frame_height=self.frame_size[1]
        x0 = int((label_width - frame_width) / 2)
        y0 = int((label_height - frame_height) / 2)
        relative_x= x-x0
        relative_y= y-y0
        self.parking_co_ordinates.append([relative_x,relative_y])
        self.update_image()
    
    def update_image(self):
        try:
            self.frame=cv2.imread(self.image_path_)
            self.frame=cv2.resize(self.frame,self.frame_size)
            for points in self.parking_co_ordinates:
                cv2.circle(self.frame, points, 1, (0,0,255), -1)
            self.rgb_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.cvt2qt = QImage(self.rgb_image.data, self.rgb_image.shape[1], self.rgb_image.shape[0], QImage.Format_RGB888)
            self.frame_.setPixmap(QPixmap.fromImage(self.cvt2qt))
        except:
            traceback.print_exc()    
            pass

    def remove_last_cordinate(self):
        if len(self.parking_co_ordinates)>0:
            self.parking_co_ordinates.pop(-1)
        self.update_image()
         
    def update_parking_data(self,settext=False,ptypelist=[]):
        self.all_parkings={"parking id":[],"parking cordindate":[],"parking type":[]}
        parking_dict= {int(i//4)+1: self.parking_co_ordinates[i:i+4] for i in range(0, len(self.parking_co_ordinates),4)}
        
        for i,keys in enumerate(parking_dict.keys()):
            self.all_parkings["parking id"].append(keys)
            self.all_parkings["parking cordindate"].append(parking_dict[keys])
            new_combox=self.parking_type_drop_down()
            if settext:
                new_combox.setCurrentText(ptypelist[i])
            self.all_parkings["parking type"].append(new_combox)
       
        parkings_in_parking_dict=list(parking_dict.keys())
        row_count = (len(parkings_in_parking_dict))
        column_count = (len(self.all_parkings.keys()))

        self.Parking_type_table_view.setColumnCount(column_count) 
        self.Parking_type_table_view.setRowCount(row_count)
        self.Parking_type_table_view.setHorizontalHeaderLabels(list(self.all_parkings.keys()))

        for row in range(row_count):  # add items from array to QTableWidget
            for col,column in enumerate(self.all_parkings.keys()):
                item = (self.all_parkings[column][row])
                if col==2:
                    self.Parking_type_table_view.setCellWidget(row, col, item)
                else:
                    item=str(item)
                    self.Parking_type_table_view.setItem(row, col, QTableWidgetItem(item))

        self.Parking_type_table_view.horizontalHeader().setStretchLastSection(True)
        self.Parking_type_table_view.resizeColumnsToContents()
        self.Parking_type_table_view.resizeRowsToContents()
    

    def parking_type_drop_down(self):
        combobox=QComboBox()
        combobox.addItems(["Red Zone","Yellow Zone","Residential","villa","Disabled","Premium","Standard"])
        return combobox
    
    def permit_type_drop_down(self):
        combobox=QComboBox()
        combobox.addItems(["Red Zone","Yellow Zone","Residential","villa","Disabled","Premium","Standard"])
        return combobox
    
    def save_parkings(self):
        for ids in self.all_parkings["parking id"]:
            parking_obj=Parking()
            parking_obj.id=int(ids)
            parking_obj.parking_type=self.all_parkings["parking type"][int(ids)-1].currentText()
            parking_obj.rects=self.all_parkings["parking cordindate"][int(ids)-1]
            
            parking_obj.xs=[x[0] for x in parking_obj.rects]
            parking_obj.ys=[x[1] for x in parking_obj.rects]
            parking_centre=(sum(parking_obj.xs)//4,sum(parking_obj.ys)//4)
            parking_obj.centroid=parking_centre

            max_xs=max(parking_obj.xs)
            min_xs=min(parking_obj.xs)
            max_ys=max(parking_obj.ys)
            min_ys=min(parking_obj.ys)

            parking_obj.top_line=[(min_xs,min_ys),(max_xs,min_ys)]
            parking_obj.bottom_line=[(min_xs,max_ys),(max_xs,max_ys)]
            parking_obj.left_line=[(min_xs,min_ys),(min_xs,max_ys)]
            parking_obj.right_line=[(max_xs,min_ys),(max_xs,max_ys)]
            self.parking_data[ids]=parking_obj


    def open_processing_window(self):
        
        # print("opening new window")
        self.Form = QtWidgets.QWidget()
        self.processing_ui = processing_window.Ui_Form()
        self.processing_ui.setupUi(self.Form)
        self.Form.setFixedWidth(self.uiscreenWidth)
        self.Form.setFixedHeight(self.uiscreenHeight-20)
        self.Form.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.processing_ui.uploadvideo.clicked.connect(self.upload_video)
        self.processing_ui.start_processing.clicked.connect(self.start_processing_)
        self.processing_ui.stop_processing.clicked.connect(self.stop_processing_)
        self.Form.show()

    def stop_processing_(self):
        self.processing_ui.video_processor.clear()
        self.car_tracker_object.stop_thread()

    def upload_video(self):
        video_path = QFileDialog.getOpenFileName(
            None, 'Test Dialog', os.getcwd(), 'All Files(*.*)')
        self.video_path_=r""+video_path[0]


    def updateFrame(self, image):
        qpix_img  = QPixmap.fromImage(image)
        self.processing_ui.video_processor.setPixmap(qpix_img.scaled(self.processing_ui.video_processor.size(),
                                            Qt.AspectRatioMode.KeepAspectRatio))
    def update_progress_bar(self,value_):
        self.processing_ui.progressBar.setProperty("value", value_)

    def start_processing_(self):
        userday=self.daySelectorCombobox.currentText()
        usertime=int(self.startTimeCombobox.currentText())
        userhour=int(self.hourCombobox.currentText())*3600
        userminute=int(self.minutesCombobox.currentText())*60
        timeMutiplier=userhour+userminute
        print("user time ===",usertime)
        self.car_tracker_object=car_tracker(self.model_obj,userday=userday,usertime=usertime,timefactor=timeMutiplier)
        self.car_tracker_object.set_values_from_server(source=self.video_path_,parking_dict=self.parking_data,pui=self.processing_ui)
        self.car_tracker_object.start()

        parking_type_list=[]
        for pt in list(self.parking_data.values()):
            parking_type_list.append(pt.parking_type)

        data_length=len(parking_type_list)

        self.parking_detail={"ID":list(self.parking_data.keys()),"parking Type":parking_type_list,
        "Status":["Empty"]*data_length,"In Time":[""]*data_length,"Out Time":[""]*data_length,"Event":[""]*data_length}
        row_count = (len(self.parking_detail["ID"]))
        column_count = (len(self.parking_detail.keys()))
        self.processing_ui.detail_table.setWordWrap(True)
        self.processing_ui.detail_table.setColumnCount(column_count) 
        self.processing_ui.detail_table.setRowCount(row_count)
        self.processing_ui.detail_table.setHorizontalHeaderLabels(list(self.parking_detail.keys()))
        self.processing_ui.detail_table.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Fixed)

        for row in range(len(self.parking_detail["ID"])):  # add items from array to QTableWidget
            for col,column in enumerate(self.parking_detail.keys()):
                if column == "Status":
                    item=QTableWidgetItem("")
                    if str(self.parking_detail[column][row])=="Occupied":
                        item.setBackground(QColor(0,255,0))
                    else:
                        item.setBackground(QColor(255,0,0))
                else:
                    item = (self.parking_detail[column][row])
                    item=str(item)
                self.processing_ui.detail_table.setItem(row, col, QTableWidgetItem(item))
                
        self.processing_ui.detail_table.horizontalHeader().setStretchLastSection(True)
        
        self.processing_ui.detail_table.resizeColumnsToContents()
        self.processing_ui.detail_table.resizeRowsToContents()

        self.car_tracker_object.progress_signal.connect(self.update_progress_bar)
        self.car_tracker_object.parking_status_signal.connect(self.update_detail_table)
        
    def update_detail_table(self,event_data):
        self.parking_detail["Status"]=[]
        for parking in self.parking_data.values():
            parking_status="Occupied" if parking.park_in_status else "Empty" 
            self.parking_detail["Status"].append(parking_status)
       
        event_key=list(event_data.keys())[0]
        event_data_=event_data[event_key]["event"]
        length = len(event_data_.split('\n'))
      
        self.parking_detail["Event"][event_key-1]+=event_data_ + "\n"
        self.parking_detail["In Time"][event_key-1]+= event_data[event_key]["ts_in"] + "\n"*length
        self.parking_detail["Out Time"][event_key-1]+= event_data[event_key]["ts_out"] + "\n"*length

        for row in range(len(self.parking_detail["ID"])):  # add items from array to QTableWidget
            for col,column in enumerate(self.parking_detail.keys()):
                if column == "Status":
                    item=QTableWidgetItem("")
                    if str(self.parking_detail[column][row])=="Occupied":
                        item.setBackground(QColor(0,255,0))
                    else:
                        item.setBackground(QColor(255,0,0))
                else:

                    item = (self.parking_detail[column][row])
                    item=str(item)

                self.processing_ui.detail_table.setItem(row, col, QTableWidgetItem(item))
        
        self.processing_ui.detail_table.horizontalHeader().setStretchLastSection(True)
        self.processing_ui.detail_table.setWordWrap(True)
        self.processing_ui.detail_table.resizeColumnsToContents()
        self.processing_ui.detail_table.resizeRowsToContents()

def main():
    app = QApplication(sys.argv)
    screen_rect = app.desktop().screenGeometry()
    screenwidth, screenheight = screen_rect.width(), screen_rect.height()-40
    win = Window(screenwidth,screenheight)
    # win.setStyleSheet(style)
    win.show()
    sys.exit(app.exec_())

if __name__ =="__main__":
    main()