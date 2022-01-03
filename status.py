import enum

class ObjStatus(enum.Enum):
    """Implementation of finite state of objects"""
    dont_know = None
    # State the object is in -> has been counted
    location_up = 0
    location_down = 1
    location_left = 2
    location_right = 3
    # Movement state of object
    movement_going_up = 4
    movement_going_down = 5
    movement_going_left = 6
    movement_going_right = 7

class TrackStatus(enum.Enum):
    """Track Option, whether to track only up/down or left/right or both"""
    none = -1
    up_down = 0
    left_right = 1
    both = 2

class Mode(enum.Enum):
    """Modes to switch between Tracking & Detection"""
    waiting = 0
    tracking = 1
    detecting = 2

if __name__ == "__main__":
    x = ObjStatus.dont_know
    print(ObjStatus.location_up)
    print(ObjStatus.movement_going_down)
    print(x is ObjStatus.movement_going_down)