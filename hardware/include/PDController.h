#ifndef PDCONTROLLER_H
#define PDCONTROLLER_H

struct ServoCommand {
    int x;
    int y;
};

class PDController {
    private:
        float frame_cx, frame_cy;
        float Kp_x, Kd_x;
        float Kp_y, Kd_y;
        float alpha;
        float deadpx;
        float max_step_x, max_step_y;
        float prev_error_x, prev_error_y;
        float prev_step_x, prev_step_y;
        float current_x, current_y;

    public:
        PDController(float frame_cx, float frame_cy,
                 float Kp_x = 0.05f, float Kd_x = 0.005f,
                 float Kp_y = 0.010f, float Kd_y = 0.004f,
                 float alpha = 0.5f, float deadpx = 12.0f,
                 float max_step_x = 0.3f, float max_step_y = 0.2f);
        
        ServoCommand update(float face_x, float face_y);
};



#endif