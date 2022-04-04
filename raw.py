import datetime
import time
frameId=0
fps=10
user_time=10*3600
previous_time=-1
while True:
    frameId+=1
    user_time_ =  user_time+int(frameId/fps)
    conversion = datetime.timedelta(seconds=user_time_)
    conversion=str(conversion)
    if "," in conversion:
        conversion=conversion.split(",")[-1].strip()
    d = datetime.datetime.strptime(str(conversion), "%H:%M:%S")
    print(d)
    current_frame_time = d.strftime("%I:%M:%S %p")
    str_current_frame = str(d.strftime("%I:%M:%S %p"))
    if previous_time ==-1:
        previous_time=current_frame_time
    else:

        print(previous_time>current_frame_time)
    print(type(str_current_frame))
    print(str_current_frame)
    time.sleep(0.1)