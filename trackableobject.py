from collections import deque
import time

from status import ObjStatus

class TrackableObject:

    # class or static variable for video_fps
    # will be required for calculating dwell time of object
    video_fps : int = 0

    def __init__(self, object_id, class_id, centroid, track_buffer, line_centre):
        # tracking buffer size
        self.track_buffer = track_buffer
        # store the object ID generated by Centroid Tracker
        self.object_id : int = object_id
        # store the object label linked by Centroid Tracker
        self.class_id : int = class_id
        # initialize a queue of centroids using the current centroid
        self.centroids = deque([centroid], maxlen=track_buffer)
        # Contains object's real time dwell time
        self.dwelltime = 0
        # Initializes the start time as soon as the object is initialized
        self.start_time = time.perf_counter()
        # Check if id has been deregistered from tracking
        self.is_deregistered = False
        # State flags
        self.status_location_X = ObjStatus.location_left if centroid[0] < line_centre[0] else ObjStatus.location_right
        self.status_location_Y = ObjStatus.location_up if centroid[1] < line_centre[1] else ObjStatus.location_down
        self.status_movement_X = ObjStatus.dont_know
        self.status_movement_Y = ObjStatus.dont_know
        # store the number of frames the object has been tracked
        self._frame_count = 1
        # stores if the object has been counted or not
        self.counted = False

    def update(self, centroid):
        """Updates the atributes TrackableObject
            1. Appends Centroid
            2. Increments frame count for specific object
            3. Computes dwell time
            4. COmputes x, y direction
        """
        self.centroids.appendleft(centroid)    # new_point -> old_point
        self._frame_count += 1
        # self._dwelltime = self._frame_count/self.video_fps

        # compute the difference between the x and y coordinates 
        # and estimate the movement direction
        # new_point - old_point
        dX = self.centroids[0][0] - self.centroids[-1][0]
        dY = self.centroids[0][1] - self.centroids[-1][1]
        self.status_movement_X = ObjStatus.movement_going_left if dX < 0 else ObjStatus.movement_going_left
        self.status_movement_Y = ObjStatus.movement_going_down if dY > 0 else ObjStatus.movement_going_up

# Singleton class
class TrackableObjects_singleton:
    # A dictionary which maps an objectID to a TrackableObject
    # Keeps id till present in frame
    # This dict is changed from TFLite_detection_track.py line 350
    # & is accessed by the counter.py Thread class for counting
    # {objectID : TrackableObject}
    _trackableObjects = {}