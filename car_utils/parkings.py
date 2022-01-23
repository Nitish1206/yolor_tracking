class Parking:
    def __init__(self) :
        self.id=None
        self.rects=None
        self.centroid=None
        self.park_in_status=False
        self.park_out_status=False
        self.parking_status=False
        self.car_number=None
        self.occupied_center=None
        self.park_in_buffer=0
        self.park_out_buffer=0
        self.park_in_time=[]
        self.park_out_time=[]
        self.xs=[]
        self.ys=[]
