from agent_states import State

class FaceTrackingAgent:
    def __init__(self, initial_state: State):
        self.state = initial_state

    def update(self, face_center):
        # Observe: allow state to request a transition
        next_state = self.state.observe(face_center) # Returns none in manual
        if next_state:
            self.state = next_state
        # Reason: compute the next goal
        goal = self.state.reason()
        return goal