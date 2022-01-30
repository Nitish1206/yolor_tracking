import cv2
import os

from torch import imag


class parking_rects:
    def __init__(self) -> None:
        self.park_rect=[]
        self.frame=None

    def click_event(self,event, x, y, flags, params):
    
        # checking for left mouse clicks
            # on the Shell
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, ' ', y)
            self.park_rect.append([x,y])
           
    def get_coordinate(self):
        image_directory=r"N:\Projects\yolor_tracking\videos\images"
        # image_name="ion_alarm.jpg"
        # image_name="Hikvision.jpg"
        image_name="image.jpg"
        image_path=os.path.join(image_directory,image_name)
        self.frame=cv2.imread(image_path)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.click_event)
        cv2.imshow('image', self.frame)
        print("rect points==",self.park_rect)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def return_area(self):
        return self.park_rect,self.frame



# img = cv2.imread(r'N:\Projects\Automatic_car_parking_system\image.jpg', 1)
# rect_obj=parking_rects()
# cord_list=rect_obj.get_coordinate(img)
# print(rect_obj.park_rect)
    