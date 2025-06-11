class FaceTrackingAgent:
    def __init__(self):
        self.current_goal = None
        self.face_lost_counter = 0
        self.lost_threshold = 30  # frames before we give up

    def observe(self, face_center):
        if face_center is None:
            self.face_lost_counter += 1
        else:
            self.face_lost_counter = 0
            self.current_goal = face_center

    def decide(self):
        if self.face_lost_counter >= self.lost_threshold:
            self.current_goal = (320, 240)  # center scan position or idle

    def act(self):
        return self.current_goal