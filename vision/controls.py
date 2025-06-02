import numpy as np

class PDController:
    def __init__(self, frame_cx, frame_cy,
                 Kp=0.05, Kd=0.01, 
                 alpha=0.2, deadpx=5,
                 max_step_x=2, max_step_y=1):
        self.frame_cx = frame_cx
        self.frame_cy = frame_cy
        self.Kp_x = Kp
        self.Kd_x = Kd
        self.Kp_y = Kp
        self.Kd_y = Kd
        self.alpha = alpha
        self.deadpx = deadpx
        self.max_step_x=max_step_x
        self.max_step_y=max_step_y

        self.prev_error_x = 0
        self.prev_error_y = 0
        self.prev_step_x = 0
        self.prev_step_y = 0
        self.current_x = 90
        self.current_y = 0


    def update(self, face_x, face_y, x_smoother, y_smoother):       
        smoothed_x = x_smoother.update(face_x)
        smoothed_y = y_smoother.update(face_y)

        # error between center of frame and face
        error_x = smoothed_x - self.frame_cx # more left = more negative
        error_y = smoothed_y - self.frame_cy # more up = more negative

        # Pixel deadzones (to account for large jitters in face detection)
        if abs(error_x) < self.deadpx: error_x = 0
        if abs(error_y) < self.deadpx: error_y = 0

        d_error_x = (error_x - self.prev_error_x) # velocity in x
        d_error_y = (error_y - self.prev_error_y) # velocity in y

        self.prev_error_x = error_x
        self.prev_error_y = error_y

        # PD control -> desired angle
        pd_cmd_x = self.Kp_x*error_x + self.Kd_x*d_error_x
        pd_cmd_y = self.Kp_y*error_y + self.Kd_y*d_error_y
        # clamp to reasonable step
        pd_cmd_x = max(-self.max_step_x, min(self.max_step_x, pd_cmd_x))
        pd_cmd_y = max(-self.max_step_y, min(self.max_step_y, pd_cmd_y))

        
        servo_step_x = self.alpha*pd_cmd_x + (1-self.alpha)*self.prev_step_x * 0.95
        servo_step_y = self.alpha*pd_cmd_y + (1-self.alpha)*self.prev_step_y * 0.95
        self.prev_step_x = servo_step_x
        self.prev_step_y = servo_step_y

        self.current_x = np.clip(self.current_x - servo_step_x, 0, 180)
        self.current_y = np.clip(self.current_y - servo_step_y, 0, 45)
        print(f"err_y: {error_y:.2f}, d_err_y: {d_error_y:.2f}, pd_y: {pd_cmd_y:.2f}, step_y: {servo_step_y:.2f}, current_y: {self.current_y:.2f}")

        return self.current_x, self.current_y