class Parking:
    def __init__(self) :
        self.id=None
        self.rects=None
        self.centroid=None
        self.park_in_status=False
        self.park_out_status=False
        self.parking_status=False
        self.occupied_center=None
        self.park_in_buffer_=0
        self.park_out_buffer=0
        self.park_in_time=[]
        self.park_out_time=[]
        self.xs=[]
        self.ys=[]
        self.car_number=[]
        self.current_car_number=None
        # parkings line segment

        self.top_line=None
        self.bottom_line=None
        self.left_line = None
        self.right_line=None

        #parking type
        self.car_id=None
        self.parking_type=None
