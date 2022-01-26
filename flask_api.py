from grpc import server
from flask import Flask,jsonify,request
from flask_cors import CORS, cross_origin
import numpy as np
import urllib.request
import requests
import json
from yolor_car_detection import car_tracker
from threading import Thread

app = Flask(__name__)
CORS(app, support_credentials=True)


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


@app.route("/start_process", methods=["POST"])
def get_data():
    data_={}
    if request.method == "POST":
        data_ = request.form.to_dict()
    
    car_tracker_obj.stop_main_process()
    parkings_from_server = json.loads(data_['placement'])
    camera_url = data_['camera_url']
    urllib.request.urlretrieve(camera_url, 'video_name.mp4')
    car_tracker_obj.set_values_from_server(source='video_name.mp4',parking_dict=parkings_from_server,server_flag=True)
    # car_tracker_obj.main()
    t1= Thread(target=car_tracker_obj.main)
    t1.start()
    return "success"

@app.route("/stop_process")
def stop_process():
    car_tracker_obj.stop_main_process()
    return "200"

@app.route("/data_recieved")
# @cross_origin(supports_credentials = True)
def reset_server_parameter():
    car_tracker_obj.reset_server_parameters()
    return "ok"

@app.route("/get_progress")
# @cross_origin(supports_credentials = True)
def send_progress():
    data={"percent": car_tracker_obj.progress}
    # print("data===",data)
    return json.dumps(data,default=default)

@app.route("/final_data",methods=["POST"])
@cross_origin(supports_credentials = True)
def send_data():

    data=car_tracker_obj.data_to_send
    print("data",data)
    return json.dumps(data,default=default)

if __name__ == '__main__' :

    car_tracker_obj = car_tracker()
    # parkings = park_obj.start()
    app.run(host="0.0.0.0", port=8080)
