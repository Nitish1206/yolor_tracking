import math
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def get_distance(point1, point2, axis):
        if axis == "x":
            return  math.sqrt((point1[0]-point2[0])**2)
        if axis == "y":
            return math.sqrt((point1[1] - point2[1]) ** 2)
        if axis == "xy":
            return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def near_parking(startX, startY, endX, endY, parking_point):
    tempX = int((startX+endX)/2)
    tempY = int((startY+endY)/2)
    car_height = endY-startY
    car_center = (tempX, tempY+(car_height//2))
    # cv2.circle(self.image, tuple(car_center), 5, (255, 0, 0), -1)
    dist_xy = get_distance(parking_point, car_center, "xy")
    return dist_xy, car_center

def if_is_inside(xs,ys, car_point,pflag):
    x_shrink = 0.7
    y_shrink = 0.3
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    # shrink figure
    new_xs = [int((i - x_center) * (1 - x_shrink) + x_center) for i in xs]
    new_ys = [int((i - y_center) * (1 - y_shrink) + y_center) for i in ys]

    # create list of new coordinates
    new_coords = [(x,y) for x,y in zip(new_xs,new_ys) ]

    point = Point(car_point)
    polygon = Polygon(new_coords)
    status=polygon.contains(point)

    return status,new_coords
    
    # if min(new_xs) < car_point[0] < max(new_xs) and min(new_ys) < car_point[1] < max(new_ys):
    #     return True ,new_coords
    # else:
    #     return False ,new_coords



