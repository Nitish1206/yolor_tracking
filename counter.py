from collections import deque
from threading import Thread
import time
import traceback

from trackableobject import TrackableObject, TrackableObjects_singleton
from status import ObjStatus, TrackStatus

class HelperCounter:
    """helper class for movement counting"""
    def __init__(self, labels) -> None:
        self.labels = labels
        self.reset()

    # Resets the counter
    def reset(self) -> None:
        # self._objects_ids_dict[label][0] -> count
        # self._objects_ids_dict[label][1] -> object_ids
        self._objects_ids_dict = dict.fromkeys(self.labels, [0 ,[]])

    # Returns the object count per label
    def get_count(self, label) -> int:
        return self._objects_ids_dict[label][0]

    # Returns total object count
    def tot_obj_count(self) -> int:
        return sum(self.get_count(label) for label in self.labels)

    # Appends object ids to list
    def add_objects(self, class_id : int, object_id : int) -> None:
        self._objects_ids_dict[self.labels[class_id]][1].append(object_id)  # append obj_ids
        self._objects_ids_dict[self.labels[class_id]][0] += 1               # increment count

    # Returns the object ids of particular movement
    def get_object_ids(self) -> list:
        return dict(filter(lambda item : item[1][0] != 0, self._objects_ids_dict.items()))
        return self._objects_ids_dict

# Singleton Thread class for counting
class Counter():
    # class variables -> shared by all objects
    left_movement = None
    right_movement = None
    up_movement = None
    down_movement = None
    dwell_time = None
    reference_line = None
    trackable_objects = TrackableObjects_singleton._trackableObjects
    run_thread = None
    track_option = None
    counter = {}
    labels = []
    demarcation_entry = {}
    demarcation_exit = {}
    
    def __init__(self, line_centre, track_option, labels) -> None:
        # initialize class variables wrt user input
        Counter.reference_line = line_centre
        Counter.run_thread = True
        Counter.track_option = track_option
        Counter.labels = labels
        Counter.left_movement = HelperCounter(Counter.labels)
        Counter.right_movement = HelperCounter(Counter.labels)
        Counter.up_movement = HelperCounter(Counter.labels)
        Counter.down_movement = HelperCounter(Counter.labels)
        Counter.counter = dict.fromkeys(Counter.labels , 0)
        Counter.dwell_time = dict.fromkeys(Counter.labels , {})
        Counter.demarcation_entry = dict.fromkeys(Counter.labels , set())
        Counter.demarcation_exit = dict.fromkeys(Counter.labels , set())

    @classmethod
    def reset_values(cls) -> None:
        cls.left_movement.reset()
        cls.right_movement.reset()
        cls.up_movement.reset()
        cls.down_movement.reset()
        cls.counter = dict.fromkeys(cls.labels , 0)
        cls.dwell_time = dict.fromkeys(cls.labels , {})

    def start(self):
        thread_obj = Thread(target=self.run)
        thread_obj.start()

    def run(self):
        while self.run_thread:
            try:
                for ((class_id, object_id), to) in Counter.trackable_objects.copy().items():
                    # find movement direction
                    # when buffer gets full or the object gets deregistered
                    to : TrackableObject
                    if (len(to.centroids) == to.track_buffer) or to.is_deregistered:
                        # up/down tracking
                        if self.track_option is TrackStatus.up_down or self.track_option is TrackStatus.both: 
                            self.check_up_down(to)
                        # left/right tracking
                        if self.track_option is TrackStatus.left_right or self.track_option is TrackStatus.both:
                            self.check_left_right(to)

                    # calculate dwell time when the object gets deregistered
                    if to.is_deregistered:
                        # calculate dwell time of particular object
                        to.dwelltime = time.perf_counter() - to.start_time
                        Counter.dwell_time[Counter.labels[class_id]][object_id] = round(to.dwelltime, 2)
                        del Counter.trackable_objects[(class_id, object_id)]

                    # check to see if the object has been counted or not
                    if not to.counted:
                        Counter.counter[Counter.labels[class_id]] += 1
                        Counter.trackable_objects[(class_id, object_id)].counted = True

            except Exception as e:
                # To check if exceptions take place inbetween thread
                print("THREAD EXCEPTION ->", e)
                print(traceback.print_exc())
                pass
            
            finally:
                time.sleep(0.01)
            
    @classmethod
    def check_left_right(cls, to : TrackableObject):
        centroid_list : deque = to.centroids
        # left movement
        if (to.status_movement_X is ObjStatus.movement_going_left and \
            to.status_location_X is ObjStatus.location_right):
            if centroid_list[0][0] < Counter.reference_line[0] < centroid_list[-1][0]:
                cls.left_movement.add_objects(to.class_id, to.object_id)
                to.status_location_X = ObjStatus.location_left

        # right movement
        elif (to.status_movement_X is ObjStatus.movement_going_right and \
              to.status_location_X is ObjStatus.location_left):
            if centroid_list[0][0] > Counter.reference_line[0] > centroid_list[-1][0]:
                cls.right_movement.add_objects(to.class_id, to.object_id)
                to.status_location_X = ObjStatus.location_right
    
    @classmethod
    def check_up_down(cls, to : TrackableObject):
        centroid_list : deque = to.centroids
        # up movement
        if (to.status_movement_Y is ObjStatus.movement_going_up and \
            to.status_location_Y is ObjStatus.location_down):
            if centroid_list[0][1] < Counter.reference_line[1] < centroid_list[-1][1]:
                cls.up_movement.add_objects(to.class_id, to.object_id)
                to.status_location_Y = ObjStatus.location_up

        # down movement
        elif (to.status_movement_Y is ObjStatus.movement_going_down and \
            to.status_location_Y is ObjStatus.location_up):
            if centroid_list[0][1] > Counter.reference_line[1] > centroid_list[-1][1]:
                cls.down_movement.add_objects(to.class_id, to.object_id)
                to.status_location_Y = ObjStatus.location_down

    def stop_thread(self):
        self.run_thread = False
