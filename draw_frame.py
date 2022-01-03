from collections import deque
from threading import Thread
import time
import cv2
import numpy as np

from utils.plots import plot_one_box
from status import TrackStatus, Mode
from trackableobject import TrackableObject, TrackableObjects_singleton

class draw_frame():
    
    def __init__(self, TRACK_OPTION, frame_height, frame_width, names, colors, TRACK_BUFFER) -> None:
        self._run_thread : bool = True
        self.frames_queue : deque = deque()
        self.trackable_objects = TrackableObjects_singleton._trackableObjects
        # initialization internal params
        self._TRACK_OPTION = TRACK_OPTION
        self._frame_height, self._frame_width = frame_height, frame_width
        self._names, self._colors = names, colors
        self._TRACK_BUFFER = TRACK_BUFFER
        # Thickness parameters
        self._tl = (0.002 * (self._frame_height + self._frame_width) / 2)   # line thickness const wrt dimensions
        self._tf = max(self._tl, 1)                                         # font thickness

    def start(self) -> object:
        thread_obj = Thread(target=self.run)
        thread_obj.start()
        return self

    def run(self):
        while self.run_thread:
            if len(self.frames_queue) > 0:
                
                # Unpack frame data from queue
                frame_data = self.frames_queue.pop()
                frame, line_centre, MODE, track_rects, scores, frame_rate_calc, \
                (counter_text, dwell_text, left_text, right_text, up_text, down_text) = frame_data

                # centre lines
                if self._TRACK_OPTION != TrackStatus.none:
                    bandwidth = 2/100
                    if self._TRACK_OPTION is TrackStatus.left_right or self._TRACK_OPTION is TrackStatus.both: 
                        # left <-> right line
                        cv2.line(frame, (line_centre[0], 0), (line_centre[0], self._frame_height), (0, 0, 255), int(self._tl))
                        # left_line_centre = int(line_centre[0] * (1-bandwidth))
                        # right_line_centre = int(line_centre[0] * (1+bandwidth))
                        # # left band
                        # cv2.line(frame, (left_line_centre, 0), (left_line_centre, self._frame_height), (50, 110, 240), 2)
                        # # right band
                        # cv2.line(frame, (right_line_centre, 0), (right_line_centre, self._frame_height), (50, 110, 240), 2)
                    if self._TRACK_OPTION is TrackStatus.up_down or self._TRACK_OPTION is TrackStatus.both: 
                        # up <-> down line
                        cv2.line(frame, (0, line_centre[1]), (self._frame_width, line_centre[1]), (0, 0, 255), int(self._tl))
                        # up_line_centre = int(line_centre[1] * (1-bandwidth))
                        # down_line_centre = int(line_centre[1] * (1+bandwidth))
                        # # left band
                        # cv2.line(frame, (0, up_line_centre), (self._frame_width, up_line_centre), (50, 110, 240), 2)
                        # # right band
                        # cv2.line(frame, (0, down_line_centre), (self._frame_width, down_line_centre), (50, 110, 240), 2)

                # Loop over the track_rects list & Draw bbox
                for class_id in range(len(self._names)):
                    if track_rects[class_id] is None:
                        continue
                    for i in range(len(track_rects[class_id])):
                        # Get bbox coordinates & label
                        xyxy = track_rects[class_id][i]
                        label = self._names[class_id]
                        if MODE is Mode.detecting:
                            text_label = '%s: %d%%' % (label, scores[class_id][i] * 100)    # Example: 'person: 72%'
                        elif MODE is Mode.tracking:
                            text_label = '%s' % (label)  # Example: 'person'
                        plot_one_box(xyxy, frame, label=text_label, color=self._colors[class_id]) # Draw bbox & label
                
                # Loop over the trackable objects dictionary & Draw centroids
                for (class_id, object_id) in self.trackable_objects.copy().keys():
                    
                    to : TrackableObject = self.trackable_objects[(class_id, object_id)]
                    
                    # compute the thickness of the line and draw the connecting lines
                    pts : deque = to.centroids
                    for i in np.arange(1, len(pts)):
                        thickness = int(np.sqrt(self._TRACK_BUFFER / float(i + 1)) * self._tl)
                        cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), self._colors[class_id], thickness)
                    
                    c1 = pts[0].copy()
                    # draw the ID of the object on the output frame
                    text = "{}_{}".format(class_id, object_id)
                    t_size = cv2.getTextSize(text, 0, fontScale=self._tl/6, thickness=int(self._tf))[0]
                    c1[0] += 5*self._tl
                    c1[1] += int(t_size[1]/2)
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(frame, c1, c2, (255, 255, 255), -1, cv2.LINE_AA)      # white filled
                    cv2.putText(frame, text, (c1[0], c1[1] - 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, self._tl/6, (0, 0, 0), int(self._tf/1.5), lineType=cv2.LINE_AA)                # draw the centroid of the object on the output frame
                    
                    cv2.circle(frame, tuple(pts[0]), int(3*self._tl), self._colors[class_id] , -1)
                    cv2.circle(frame, tuple(pts[0]), int(2*self._tl), (0, 0, 0) , -1)
                
                # Display FPS & count
                y = 80
                k = int(10*self._tl)
                # Write data to frame
                cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), 
                            (30, y), cv2.FONT_HERSHEY_SIMPLEX, int(self._tl/3), (0, 0, 255), int(self._tf), cv2.LINE_AA)
                # Draw count in frame
                cv2.putText(frame, "Counter : " + counter_text, 
                            (30, y+2*k), cv2.FONT_HERSHEY_SIMPLEX, int(self._tl/3), (0, 0, 255), int(self._tf), cv2.LINE_AA)
                cv2.putText(frame, "Dwell time: " + dwell_text, 
                            (30, y+3*k), cv2.FONT_HERSHEY_SIMPLEX, int(self._tl/3), (0, 0, 255), int(self._tf), cv2.LINE_AA)
                if self._TRACK_OPTION != TrackStatus.none:
                    if self._TRACK_OPTION is TrackStatus.left_right or self._TRACK_OPTION is TrackStatus.both: 
                        cv2.putText(frame, "Right: " + right_text,
                                    (30, y+4*k), cv2.FONT_HERSHEY_SIMPLEX, int(self._tl/3), (0, 0, 255), int(self._tf), cv2.LINE_AA)
                        cv2.putText(frame, "Left: " + left_text, 
                                    (30, y+5*k), cv2.FONT_HERSHEY_SIMPLEX, int(self._tl/3), (0, 0, 255), int(self._tf), cv2.LINE_AA)
                    if self._TRACK_OPTION is TrackStatus.up_down or self._TRACK_OPTION is TrackStatus.both: 
                        cv2.putText(frame, "up: " + up_text, 
                                    (30, y+6*k), cv2.FONT_HERSHEY_SIMPLEX, int(self._tl/3), (0, 0, 255), int(self._tf), cv2.LINE_AA)
                        cv2.putText(frame, "Down: " + down_text, 
                                (30, y+6*k), cv2.FONT_HERSHEY_SIMPLEX, int(self._tl/3), (0, 0, 255), int(self._tf), cv2.LINE_AA)

            time.sleep(0.01)

    def update(self, frame, line_centre, MODE, track_rects, scores, frame_rate_calc, *counter_text):
        self.frames_queue.appendleft([frame, MODE, track_rects, scores, frame_rate_calc, *counter_text])

    def stop_thread(self):
        self._run_thread = False