import math
import time
from collections import deque

# Parent class
class Task:
    """
    Abstract base class for execution primitives.
    """
    def observe(self, frame_input):
        """
        Process perception input (e.g., face_center). 
        """
        pass

    def reason(self):
        """
        Compute and return a goal (x, y) tuple.
        """
        return (0, 0)

    def is_done(self):
        """
        Return True when task is complete.
        """
        return True


class Track(Task):
    def __init__(self, lost_threshold: int = 30):
        self.face_lost_counter = 0
        self.lost_threshold = lost_threshold
        self.current_goal = (320, 240)

    def observe(self, frame_input):
        if frame_input is None:
            self.face_lost_counter += 1
        else:
            self.face_lost_counter = 0
            self.current_goal = frame_input

    def reason(self):
        return self.current_goal
    
    def is_done(self):
        # Task completes if face is lost for too long
        return self.face_lost_counter >= self.lost_threshold


class Scan(Task):
    def __init__(self, duration: float):
        self.duration = duration
        self.start_time = None
        self.idle_phase = 0.0

    def observe(self, frame_input):
        # Initialize start time on first call
        if self.start_time is None:
            self.start_time = time.time()

        # If a face is detected during scan, we could optionally finish early
        if frame_input is not None:
            # Do nothing special; continuing sweep
            pass

    def reason(self):
        # Sweep horizontally with a sine wave
        self.idle_phase += 0.01
        sweep_x = int(320 + 250 * math.sin(self.idle_phase))
        sweep_y = 240
        return (sweep_x, sweep_y)

    def is_done(self):
        return (time.time() - self.start_time) >= self.duration

# class Manual(State):
#     def __init__(self):
#         self.override_goal = (320, 240)

#     def observe(self, face_center):
#         # Ignore vision in manual mode
#         return None

#     def reason(self):
#         return self.override_goal

#     def set_goal(self, new_goal):
#         self.override_goal = new_goal


