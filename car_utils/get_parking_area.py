import cv2

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
        self.frame=cv2.imread(r"N:\Projects\yolor_tracking\videos\image.jpg")
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
    