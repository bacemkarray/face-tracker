import math
class FaceTrackingAgent:
    def __init__(self):
        self.current_goal = (320, 240)
        self.face_lost_counter = 0
        self.lost_threshold = 30  # frames before we give up
        self.mode = "idle"
        self.idle_phase = 0

    def observe(self, face_center):
        if face_center is None:
            self.face_lost_counter += 1
        else:
            self.face_lost_counter = 0
            self.current_goal = face_center
            self.mode = "tracking"

    def decide(self):
        if self.face_lost_counter >= self.lost_threshold:
            self.mode = "searching"
            self.idle_phase += 0.01  # increase slowly
            sweep_x = int(320 + 250 * math.sin(self.idle_phase))
            sweep_y = 240  # keep vertical center fixed, or slowly oscillate too
            self.current_goal = (sweep_x, sweep_y)

    def act(self):
        return self.current_goal