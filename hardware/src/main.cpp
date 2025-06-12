#include <Arduino.h>
#include <ESP32Servo.h>
#include "PDController.h"
#include "InverseK.h"

// ——— Face-tracker PD constants ———
static const float FRAME_WIDTH  = 640.0f;
static const float FRAME_HEIGHT = 480.0f;

static const float Kp_x  = 0.06f;
static const float Kd_x  = 0.007f;
static const float Kp_y  = 0.04f;
static const float Kd_y  = 0.005f;

static const float DEADPX     = 25.0f;
static const float MAX_STEP_X = 0.5f;
static const float MAX_STEP_Y = 0.5f;

// ——— Instantiate your PDController ———
PDController controller(
  FRAME_WIDTH/2, FRAME_HEIGHT/2,
  Kp_x, Kd_x,
  Kp_y, Kd_y,
  DEADPX,
  MAX_STEP_X, MAX_STEP_Y
);

// ——— Base (pan) servo ———
Servo baseServo;
static const int PIN_BASE = 12;

// ——— Inverse-Kinematics setup ———
static const float LINK1_LEN = 10.5f;   // cm
static const float LINK2_LEN = 16.5f;   // cm

static const int PIN_SHOULDER     = 21;
static const int PIN_ELBOW        = 19;
static const int SHOULDER_OFFSET  = 10;  // your calibration
static const int ELBOW_OFFSET     = 45;  // your calibration

InverseK ik(
  LINK1_LEN, LINK2_LEN,
  PIN_SHOULDER, PIN_ELBOW,
  SHOULDER_OFFSET, ELBOW_OFFSET
);

// ——— “Home” coordinates to keep X,Z fixed ———
// Measure these once (e.g. with your previous Kinematics.getPositions)
// after you’ve moved the arm to your neutral pose.
static const float FIXED_X = 19.1f;  // example: (L1+L2)*cos45°
static const float FIXED_Z = 19.1f;  // example: (L1+L2)*sin45°

void setup() {
  Serial.begin(115200);

  // pan
  baseServo.attach(PIN_BASE);

  // shoulder/elbow IK
  baseServo.write(100);
  ik.begin();
  ik.home(45, 45);
}

void loop() {
  // 1) Read face coords (4 bytes little-endian)
  if (Serial.available() < 4) return;
  uint16_t face_x = Serial.read() | (Serial.read() << 8);
  uint16_t face_y = Serial.read() | (Serial.read() << 8);

  // 2) PD update → pan angle and ΔZ
  auto cmd = controller.update((float)face_x, (float)face_y);
  baseServo.write(cmd.x);

  // 3) IK for tilt: keep X fixed, add vertical offset
  ik.moveTo(FIXED_X, FIXED_Z + cmd.y);
}