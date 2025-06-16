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


class Tracking(State):
    def __init__(self, lost_threshold=30):
        self.face_lost_counter = 0
        self.lost_threshold = lost_threshold
        self.current_goal = (320, 240)

    def observe(self, face_center):
        if face_center is None:
            self.face_lost_counter += 1
            if self.face_lost_counter >= self.lost_threshold:
                return Searching()
        else:
            self.face_lost_counter = 0
            self.current_goal = face_center
        return None

    def reason(self):
        return self.current_goal


class Searching(State):
    def __init__(self):
        self.face_lost_counter = 0
        self.idle_phase = 0.0

    def observe(self, face_center):
        if face_center is not None:
            return Tracking()
        else:
            self.face_lost_counter += 1
        return None

    def reason(self):
        self.idle_phase += 0.01
        sweep_x = int(320 + 250 * math.sin(self.idle_phase))
        sweep_y = 240
        return (sweep_x, sweep_y)


class Manual(State):
    def __init__(self):
        self.override_goal = (320, 240)

    def observe(self, face_center):
        # Ignore vision in manual mode
        return None

    def reason(self):
        return self.override_goal

    def set_goal(self, new_goal):
        self.override_goal = new_goal