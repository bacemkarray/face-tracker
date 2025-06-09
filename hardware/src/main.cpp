#include <Arduino.h>
#include <ESP32Servo.h>
#include "PDController.h"

Servo base;
Servo servo_elbow;
Servo shoulder;

// YOLO model frame dimensions
static const float FRAME_WIDTH  = 640.0f;
static const float FRAME_HEIGHT = 480.0f;

// PD gains and other constants:
static const float  Kp_x       = 0.094f;
static const float  Kd_x       = 0.00f;
static const float  Kp_y       = 0.10f;
static const float  Kd_y       = 0.00f;

// Exponential smoothing factor and other PDController params:
static const float  ALPHA      = 1.0f;    // same α that PDController uses
static const float  DEADPX     = 20.0f;   // pixels dead‐zone
static const float  MAX_STEP_X = 0.5f;    // deg/frame x
static const float  MAX_STEP_Y = 0.5f;    // deg/frame y

// Instantiate PDController:
PDController controller(
  FRAME_WIDTH/2.0f,       // frame_cx
  FRAME_HEIGHT/2.0f,      // frame_cy
  Kp_x, Kd_x,             // Kp_x, Kd_x
  Kp_y, Kd_y,             // Kp_y, Kd_y
  ALPHA,
  DEADPX,                 // dead‐zone
  MAX_STEP_X,             // max step x
  MAX_STEP_Y              // max step y
);

void setup() {
  Serial.begin(115200);

  // Attach each servo to its pin:
  base.attach(12);         // yaw / pan
  servo_elbow.attach(19); // elbow “bend”
  shoulder.attach(21);         // shoulder pitch

  // Initialize to neutral positions:
  base.write(90);
  servo_elbow.write(0);        
  shoulder.write(45);
}

void loop() {
  // We expect 4 bytes from Serial per frame (face_x, face_y as two‐byte integers):
  if (Serial.available() >= 4) {
    // Read face_x (low‐byte, high‐byte)
     uint16_t face_x = Serial.read();
    face_x |= ( (uint16_t)Serial.read() << 8 );

    // Read face_y (low‐byte, high‐byte)
    uint16_t face_y = Serial.read();
    face_y |= ( (uint16_t)Serial.read() << 8 );

    // Run PD update. Returns { x = pan, y = tilt }
    auto command = controller.update((float)face_x, (float)face_y);
    int send_x = command.x;  // “pan” angle
    int send_y = command.y;  // “tilt” angle

    base.write(send_x);             // yaw/pan stays unchanged
    servo_elbow.write(send_y); // new elbow “bend” angle
    // shoulder.write(write_shldr);         // new shoulder pitch angle
  }
}
