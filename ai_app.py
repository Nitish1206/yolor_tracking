from application.windows import roi_window,video_player,processing_window
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap,QColor
from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog,QTableWidgetItem,QComboBox
from PyQt5 import QtWidgets
import sys
import cv2
import os
from car_utils.parkings import Parking
from car_utils.cars import car
from random import randint
import torch
# from video_thread import videoThread
# from yolor_car_detection import car_tracker
from yolor_car_detection_qthread import car_tracker
from glob import glob
import traceback
import json 

cwd=os.getcwd()

def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)


class Window(QMainWindow, roi_window.Ui_MainWindow,processing_window.Ui_Form):
    def __init__(self, parent=None):
        super(Window, self).__init__()
        self.setupUi(self)
        self.setFixedWidth(1780)
        self.setFixedHeight(900)
        self.frame_size=(1080,720)
        with torch.no_grad():

            self.car_tracker_object=car_tracker()

        self.all_parkings={"parking id":[],"parking cordindate":[],"parking type":[]}
        self.parking_data={}
        self.car_data={}
        self.parking_co_ordinates=[]
        self.config_path=cwd+os.sep+"application"+os.sep+"configuration"
        self.select_config_files()
        self.upload_button.clicked.connect(self.upload_image_)
        self.clear_button.clicked.connect(self.remove_last_cordinate)
        self.save_button.clicked.connect(self.update_parking_data)
        self.save_parking.clicked.connect(self.save_parkings)
        self.frame_.mousePressEvent = self.getPixel
        self.enter_button_number_of_car.clicked.connect(self.set_cars)
        self.save_car_properties.clicked.connect(self.save_cars)
        self.videoprocessing.clicked.connect(self.open_processing_window)
        self.preconfigured.clicked.connect(self.check_if_user_selectected_precofiguration)
        self.save_config.clicked.connect(self.save_configuration)

    def check_if_user_selectected_precofiguration(self):
        if self.preconfigured.isChecked():
            precofig=True
        else:
            preconfig=False

    def select_config_files(self):
        config_files=glob(self.config_path+"/*")
        file_names=[]

        for files in config_files:
            file_names.append(os.path.basename(files))
        
        self.select_config_combobox.addItems(file_names)
   
    def save_configuration(self):
        try:
            current_config = {}
            current_config["image_path"]=self.image_path_
            all_parkings_={"parking id":[],"parking cordindate":[],"parking type":[]}
            for parking in self.parking_data.values():
                all_parkings_["parking id"].append(parking.id)
                all_parkings_["parking cordindate"].append(parking.rects)
                all_parkings_["parking type"].append(parking.parking_type)

            current_config["parking_properties"]=all_parkings_

            all_cars_={"car id":[],"car number":[],"permit type":[]}
            for car in self.car_data.values():
                all_cars_["car id"].append(car.id)
                all_cars_["car number"].append(car.number)
                all_cars_["permit type"].append(car.permit_type)


            current_config["car_properties"]=all_cars_

            save_path=self.config_path+os.sep+os.path.basename(self.image_path_).split(".")[0]
            with open(save_path+".json", "w") as outfile:
                json.dump(current_config, outfile)
        except:
            traceback.print_exc()
            pass
        

    def upload_image_(self):
        image_path = QFileDialog.getOpenFileName(
            None, 'Test Dialog', os.getcwd(), 'All Files(*.jpg*)')
        self.image_path_=r""+image_path[0]
        try:
            self.frame=cv2.imread(self.image_path_)
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
         
    def update_parking_data(self):
        self.all_parkings={"parking id":[],"parking cordindate":[],"parking type":[]}
        parking_dict= {int(i//4)+1: self.parking_co_ordinates[i:i+4] for i in range(0, len(self.parking_co_ordinates),4)}
        
        for keys in parking_dict.keys():
            self.all_parkings["parking id"].append(keys)
            self.all_parkings["parking cordindate"].append(parking_dict[keys])
            new_combox=self.parking_type_drop_down()
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

        self.Parking_type_table_view.resizeColumnsToContents()
        self.Parking_type_table_view.resizeRowsToContents()
    

    def parking_type_drop_down(self):
        combobox=QComboBox()
        combobox.addItems(["Premium","Residential","villa"])
        return combobox
    
    def permit_type_drop_down(self):
        combobox=QComboBox()
        combobox.addItems(["Premium","Residential","villa"])
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

    def set_cars(self):
        self.all_cars={"car id":[],"car number":[],"permit type":[]}
        car_length=int(self.car_numeber_value.text())
        for i in range(1,car_length+1):
            car_number = random_with_N_digits(6)
            self.all_cars["car id"].append(i)
            self.all_cars["car number"].append(car_number)
            new_permit=self.permit_type_drop_down()
            self.all_cars["permit type"].append(new_permit)

        row_count = (len(self.all_cars["car id"]))
        column_count = (len(self.all_cars.keys()))

        self.car_properites.setColumnCount(column_count) 
        self.car_properites.setRowCount(row_count)
        self.car_properites.setHorizontalHeaderLabels(list(self.all_cars.keys()))

        for row in range(row_count):  # add items from array to QTableWidget
            for col,column in enumerate(self.all_cars.keys()):
                item = (self.all_cars[column][row])
                if col==2:
                    self.car_properites.setCellWidget(row, col, item)
                else:
                    item=str(item)
                    self.car_properites.setItem(row, col, QTableWidgetItem(item))

        self.car_properites.resizeColumnsToContents()
        self.car_properites.resizeRowsToContents()
    
    def save_cars(self):

        for ids in self.all_cars["car id"]:
            car_obj=car()
            car_obj.id=int(ids)
            car_obj.permit_type=self.all_cars["permit type"][int(ids)-1].currentText()
            car_obj.number=self.all_cars["car number"][int(ids)-1]
            self.car_data[ids]=car_obj

    def open_processing_window(self):
        
        print("opening new window")
        # self.close()
        self.Form = QtWidgets.QWidget()
        # self.Form = QtWidgets.QMainWindow()
        self.processing_ui = processing_window.Ui_Form()
        self.processing_ui.setupUi(self.Form)
        self.Form.setFixedWidth(1780)
        self.Form.setFixedHeight(900)
        self.processing_ui.uploadvideo.clicked.connect(self.upload_video)
        # self.uploadvideo.clicked.connect(self.upload_video)
        self.processing_ui.start_processing.clicked.connect(self.start_processing_)
        # self.start_processing.clicked.connect(self.start_processing_)
        self.processing_ui.stop_processing.clicked.connect(self.stop_processing_)
        # self.stop_processing.clicked.connect(self.stop_processing_)

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
        # set a scaled pixmap to a w x h window keeping its aspect ratio 
        self.processing_ui.video_processor.setPixmap(qpix_img.scaled(self.processing_ui.video_processor.size(),
                                            Qt.AspectRatioMode.KeepAspectRatio))
    def update_progress_bar(self,value_):
        self.processing_ui.progressBar.setProperty("value", value_)

    def start_processing_(self):
        self.car_tracker_object.set_values_from_server(source=self.video_path_,parking_dict=self.parking_data,
        car_data=self.car_data,pui=self.processing_ui)
        
        self.car_tracker_object.start()
        self.parking_detail={"ID":list(self.parking_data.keys()),"Status":["Empty","Empty","Empty"],"Time stamp":["","",""],"Event":["","",""]}
        # self.car_tracker_object.image_signal.connect(self.updateFrame)
        row_count = (len(self.parking_detail["ID"]))
        column_count = (len(self.parking_detail.keys()))
        self.processing_ui.detail_table.setWordWrap(True)
        self.processing_ui.detail_table.setColumnCount(column_count) 
        self.processing_ui.detail_table.setRowCount(row_count)
        self.processing_ui.detail_table.setHorizontalHeaderLabels(list(self.parking_detail.keys()))

        for row in range(len(self.parking_detail["ID"])):  # add items from array to QTableWidget
            for col,column in enumerate(self.parking_detail.keys()):
               
                item = (self.parking_detail[column][row])
                item=str(item)
                self.processing_ui.detail_table.setItem(row, col, QTableWidgetItem(item))

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
      
        self.parking_detail["Event"][event_key-1]+= event_data[event_key]["event"] + "\n"
        self.parking_detail["Time stamp"][event_key-1]+= event_data[event_key]["ts"] + "\n"
        

        for row in range(len(self.parking_detail["ID"])):  # add items from array to QTableWidget
            for col,column in enumerate(self.parking_detail.keys()):
               
                item = (self.parking_detail[column][row])
                item=str(item)
                self.processing_ui.detail_table.setItem(row, col, QTableWidgetItem(item))

        self.processing_ui.detail_table.resizeColumnsToContents()
        self.processing_ui.detail_table.resizeRowsToContents()

def main():
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())

    # try:
    #     sys.exit(app.exec_())
    # except SystemExit:
    #     print("Closing Application ...")

if __name__ == '__main__':
    main()