import datetime
from pickle import TRUE
import os

seconds_=3600
def convert_seconds_to_time(user_time_):
    conversion = datetime.timedelta(seconds=user_time_)
    conversion=str(conversion)
    if "," in conversion:
        conversion=conversion.split(",")[-1].strip()
    d = datetime.datetime.strptime(str(conversion), "%H:%M:%S")
    current_frame_time = d.strftime("%I:%M:%S %p")
    return current_frame_time



RESIDENTIAL_PARKING_NO_PERMIT_NO_FINE_TIME_START = 8*seconds_
RESIDENTIAL_PARKING_NO_PERMIT_NO_FINE_TIME_START_dt=convert_seconds_to_time(RESIDENTIAL_PARKING_NO_PERMIT_NO_FINE_TIME_START)
RESIDENTIAL_PARKING_NO_PERMIT_NO_FINE_TIME_END = 21*seconds_
RESIDENTIAL_PARKING_NO_PERMIT_NO_FINE_TIME_END_dt=convert_seconds_to_time(RESIDENTIAL_PARKING_NO_PERMIT_NO_FINE_TIME_END)
RESIDENTIAL_PARKING_NO_PERMIT_FINE_TIME_START = 21*seconds_
RESIDENTIAL_PARKING_NO_PERMIT_FINE_TIME_START_dt=convert_seconds_to_time(RESIDENTIAL_PARKING_NO_PERMIT_FINE_TIME_START)
RESIDENTIAL_PARKING_NO_PERMIT_FINE_TIME_END = 8*seconds_
RESIDENTIAL_PARKING_NO_PERMIT_FINE_TIME_END_dt=convert_seconds_to_time(RESIDENTIAL_PARKING_NO_PERMIT_FINE_TIME_END)
PREMIMUM_PARKING_NO_PERMIT_NO_FINE_TIME_START = 0*seconds_
PREMIMUM_PARKING_NO_PERMIT_NO_FINE_TIME_START_dt=convert_seconds_to_time(PREMIMUM_PARKING_NO_PERMIT_NO_FINE_TIME_START)
PREMIMUM_PARKING_NO_PERMIT_NO_FINE_TIME_END = 8*seconds_
PREMIMUM_PARKING_NO_PERMIT_NO_FINE_TIME_END_dt=convert_seconds_to_time(PREMIMUM_PARKING_NO_PERMIT_NO_FINE_TIME_END)
PREMIMUM_PARKING_NO_PERMIT_FINE_TIME_START = 8*seconds_
PREMIMUM_PARKING_NO_PERMIT_FINE_TIME_START_dt=convert_seconds_to_time(PREMIMUM_PARKING_NO_PERMIT_FINE_TIME_START)
PREMIMUM_PARKING_NO_PERMIT_FINE_TIME_END = 0*seconds_  
PREMIMUM_PARKING_NO_PERMIT_FINE_TIME_END_dt=convert_seconds_to_time(PREMIMUM_PARKING_NO_PERMIT_FINE_TIME_END)


#camera view

FRONT_CAMERA=True
TOP_CAMERA=False
DRAW_STATUS=True

PERMIT_TYPES=["Residential","villa","Disabled","Premium","Standard"]

DAY_LIST=["workday","offday"]
STARTTIMES=[str(0),str(8),str(21)]
HOURDATA=[str(x) for x in range(0,24)]
MINUTESDATA=[str(x) for x in range(0,60)]

cwd=os.getcwd()
CONFIG_PATH = cwd + os.sep+"application"+os.sep+"configuration"