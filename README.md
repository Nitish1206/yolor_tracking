# CCTVAI

### Project Description
CCTV cameras or VMS to identify objects and send this information to a shared artificial intelligence cloud server to improve accuracy.

### Pipeline
Yolov5l -> *https://github.com/ultralytics/yolov5.git*
YoloR -> *https://github.com/WongKinYiu/yolor.git*

**[Project documentation](https://docs.google.com/document/d/1gaQy75o2spx3r_3Y3yRnP9q6LdDvMagQaSzTqSDrnnQ/edit?usp=sharing)**

**[Github Repo](https://github.com/Cmaktech/CCTVAI)**
**[Bitbucket Repo](https://Shuvam-WebO@bitbucket.org/chiragtweboccult/cctvai.git)**

**Object Detection Classes**:
1. Person
2. Dog
3. Cat
4. Car
5. Bicycle
6. Motorcycle
7. Boat

**Installation**
```
conda create -n yolo_detection python=3.8
conda activate yolo_detection
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch #For cuda 10.2
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge #For cuda 11.1
pip install -r requirements.txt      
```

**RUN**
```
python detect_yolor.py --agnostic-nms --device=0 --source N:\Projects\yolor_tracking\videos\test3.mp4
```