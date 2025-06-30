import math
import time

# Parent class
class Task:
    """
    Abstract base class for execution primitives.
    """
    def __init__(self):
        self.task_id = -1

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
        super().__init__()
        self.task_id = 0
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
        super().__init__()
        self.task_id = 1
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
        self.idle_phase += 0.03
        sweep_x = int(90 + 80 * math.sin(self.idle_phase))
        sweep_y = 0
        return (sweep_x, sweep_y)

    def is_done(self):
        return (time.time() - self.start_time) >= self.duration