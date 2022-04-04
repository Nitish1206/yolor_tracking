
from utils.torch_utils import select_device
from models.models import *
import ast
import colorsys

class LoadModel:
    def __init__(self):
        self.img_size=1280
        self.auto_size=64
        self.device_cfg='0' #  default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        self.cfg='cfg/yolor_p6.cfg'#, help='*.cfg path')
        self.weights=['yolo_custom.pt']#, help='model.pt path(s)')
        self.names_path='data/coco.names'#, help='*.cfg path')
        self.classes='classes.txt' #, help='classes path -> [0, 1, 3, 7]')

        self.augment=False #', 
        self.conf_thres=0.4 #', type=float, default=0.4, help='object confidence threshold')
        self.iou_thres = 0.5 #', type=float, default=0.5, help='IOU threshold for NMS')
        self.agnostic_nms = True #', action='store_true', help='class-agnostic NMS')

        self.device = select_device(self.device_cfg)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = Darknet(self.cfg, self.img_size).cuda()
        self.model.load_state_dict(torch.load(self.weights[0], map_location=self.device)['model'])
        
        self.model.to(self.device).eval()
        if self.half:
            self.model.half()

        img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
        self.names : np.ndarray = np.array(self.load_classes(self.names_path))
        self.desired_classes : list = ast.literal_eval(open( self.classes).read().strip())
        assert isinstance(self.desired_classes, list), "classes.txt does not contain a list, unsupported type -> " + type(self.desired_classes)
        self.colors : np.ndarray = np.array([None] * len(self.names))
        
        desired_colors : list = [tuple(np.array(colorsys.hsv_to_rgb(hue,1,1))[::-1]*255) 
                                        for hue in np.arange(0, 1, 1/len(self.desired_classes))]
        self.colors[self.desired_classes] = desired_colors
   
    def load_classes(self,path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  # filter removes empty strings (such as last line)