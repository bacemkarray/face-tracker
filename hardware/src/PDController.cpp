#include "PDController.h"
#include <math.h>

PDController::PDController(float f_cx, float f_cy,
                           float Kpx, float Kdx,
                           float Kpy, float Kdy,
                           float a, float dpx,
                           float mx, float my)
  : frame_cx(f_cx),
    frame_cy(f_cy),
    Kp_x(Kpx),
    Kd_x(Kdx),
    Kp_y(Kpy),
    Kd_y(Kdy),
    alpha(a),
    deadpx(dpx),
    max_step_x(mx),
    max_step_y(my),
    
    prev_error_x(0.0f),
    prev_error_y(0.0f),
    prev_step_x(0.0f),
    prev_step_y(0.0f),
    current_x(90.0f),
    current_y(0.0f)
{}

ServoCommand PDController::update(float face_x, float face_y) {
    // error between center of frame and face
    float error_x = face_x - frame_cx; // more left = more negative
    float error_y = face_y - frame_cy; // more up = more negative

    // Pixel deadzones (to prevent servo response from small face movements)
    if (fabsf(error_x) < deadpx) error_x = 0.0f;
    if (fabsf(error_y) < deadpx) error_y = 0.0f;

    // Derivative (pixels/frame)
    float d_error_x = error_x - prev_error_x;
    float d_error_y = error_y - prev_error_y;
    
    // Compute PD only if error != 0; otherwise force 0
    float pd_cmd_x;
    float pd_cmd_y;
    if(error_x == 0.0f) {
        pd_cmd_x = 0.0f;
        prev_step_x = 0.0f;
    }
    else {
        pd_cmd_x = Kp_x * error_x + Kd_x * d_error_x;
        pd_cmd_x = fminf(fmaxf(pd_cmd_x, -max_step_x), max_step_x);
    }

    if(error_y == 0.0f) {
        pd_cmd_y = 0.0f;
        prev_step_y = 0.0f;
    }
    else {
        pd_cmd_y = Kp_y * error_y + Kd_y * d_error_y;
        pd_cmd_y = fminf(fmaxf(pd_cmd_y, -max_step_y), max_step_y);
    }


    prev_error_x = error_x;
    prev_error_y = error_y;

    float servo_step_x = alpha * pd_cmd_x + (1 - alpha) * prev_step_x;
    float servo_step_y = alpha * pd_cmd_y + (1 - alpha) * prev_step_y;

    prev_step_x = servo_step_x;
    prev_step_y = servo_step_y;

    current_x = fminf(fmaxf(current_x - servo_step_x, 0.0f), 180.0f);
    current_y = fminf(fmaxf(current_y - servo_step_y, 0.0f), 45.0f);

    int send_x = (int)(roundf(current_x));
    int send_y = (int)(roundf(current_y));

    return { send_x, send_y };
}