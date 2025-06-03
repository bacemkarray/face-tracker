#include <Arduino.h>
#include <Servo.h>
#include "PDController.h"

Servo base;
Servo servo_elbow;
Servo arm;

// YOLO model frame dimensions
static const float FRAME_WIDTH  = 640.0f;
static const float FRAME_HEIGHT = 480.0f;

// PD gains and other constants:
static const float  Kp_x       = 0.05f;
static const float  Kd_x       = 0.005f;
static const float  Kp_y       = 0.05f;
static const float  Kd_y       = 0.004f;

// Exponential smoothing factor and other PDController params:
static const float  ALPHA      = 0.6f;    // same α that PDController uses
static const float  DEADPX     = 20.0f;   // pixels dead‐zone
static const float  MAX_STEP_X = 0.3f;    // deg/frame x
static const float  MAX_STEP_Y = 0.2f;    // deg/frame y

// “Neutral” (rest) angles for shoulder and elbow:
static const int    SHOULDER_MID = 90;
static const int    ELBOW_MID    = 60;

// How much of the total tilt (send_y) goes to shoulder vs. elbow:
static const float  FRAC_SHLDR = 0.6f;    // 60% of motion on shoulder
static const float  FRAC_ELBOW = 0.4f;    // 40% on elbow

// Instantiate PDController with same α:
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
  base.attach(9);         // yaw / pan
  servo_elbow.attach(10); // elbow “bend”
  arm.attach(11);         // shoulder pitch

  // Initialize to neutral positions:
  base.write(90);                      // face centered
  servo_elbow.write(ELBOW_MID);        // elbow at neutral
  arm.write(SHOULDER_MID);             // shoulder at neutral
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

    // Run PD update → returns { x = pan, y = tilt }
    auto command = controller.update((float)face_x, (float)face_y);
    int send_x = command.x;  // “pan” angle
    int send_y = command.y;  // “tilt” angle

    // ─────────── Option 1: Split‐the‐tilt between shoulder & elbow ───────────

    // 1) Compute raw (unsmoothed) targets for each joint:
    int raw_shldr = SHOULDER_MID + int((send_y - SHOULDER_MID) * FRAC_SHLDR);
    int raw_elbow = ELBOW_MID    + int((send_y - ELBOW_MID)    * FRAC_ELBOW);

    // 2) Clamp each to valid servo range [0..180]:
    raw_shldr = constrain(raw_shldr, 0, 180);
    raw_elbow = constrain(raw_elbow, 0, 180);

    // 3) Exponential smoothing (use same α that PDController is using):
    static float prev_shldr = (float)SHOULDER_MID;
    static float prev_elbow = (float)ELBOW_MID;
    float s_shldr = ALPHA * raw_shldr + (1.0f - ALPHA) * prev_shldr;
    float s_elbow = ALPHA * raw_elbow + (1.0f - ALPHA) * prev_elbow;
    prev_shldr = s_shldr;
    prev_elbow = s_elbow;

    int write_shldr = int(roundf(s_shldr));
    int write_elbow = int(roundf(s_elbow));

    // 4) Now send to hardware:
    base.write(send_x);             // yaw/pan stays unchanged
    servo_elbow.write(write_elbow); // new elbow “bend” angle
    arm.write(write_shldr);         // new shoulder pitch angle
  }
}
