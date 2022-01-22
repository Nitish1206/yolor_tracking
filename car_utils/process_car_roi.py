from threading import Thread
from helper import *

class process_park_roi(Thread):
    def __init__(self) -> None:
        self.cars=[]
        self.parkings=[]
        self.thread_status=True

    def run(self):
        while self.thread_status:
            current_parkings=self.parkings
            current_cars=self.cars
            for parking in current_parkings:
                current_parking = parking.rects
                for car in current_cars:
                    car_centroid=car.centroid
                    park_status = if_is_inside(current_parking[0],current_parking[1],current_parking[2],current_parking[3],car_centroid)
                    if park_status:
                        car.parking_id=parking.id
                        car.parked=True
    
    def stop_thread(self):
        self.thread_status=False






