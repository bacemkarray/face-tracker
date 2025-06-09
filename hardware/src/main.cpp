#include <Arduino.h>
#include <ESP32Servo.h>
#include "PDController.h"
#include <Kinematics.h>

Servo base;
Servo servo_elbow;
Servo shoulder;

// YOLO model frame dimensions
static const float FRAME_WIDTH  = 640.0f;
static const float FRAME_HEIGHT = 480.0f;

// PD gains and other constants:
static const float  Kp_x       = 0.06f;
static const float  Kd_x       = 0.007f;
static const float  Kp_y       = 0.04f;
static const float  Kd_y       = 0.005f;

// Exponential smoothing factor and other PDController params:
static const float  DEADPX     = 25.0f;   // pixels dead‐zone
static const float  MAX_STEP_X = 0.5f;    // deg/frame x
static const float  MAX_STEP_Y = 0.5f;    // deg/frame y

static const float SHOULDER_LEN = 10.5f;   // e.g. 10 cm
static const float  ELBOW_LEN   = 16.5f;   // e.g. 10 cm

static const int SHOULDER_MIN = 45;
static const int SHOULDER_MAX = 80;

static const int ELBOW_MIN = 0;
static const int ELBOW_MAX = 90;

// how many real‐world units per pixel in Y.
// Calibrate by measuring how many cm the end‐effector moves
// when the face moves N pixels vertically.
static const float Y_SCALE = 0.0286f;   // estimated 70 pixels = 2 cm

// --- globals for IK ---
Kinematics ik(SHOULDER_LEN, ELBOW_LEN);
Position  homePos;
float     fixedX, fixedZ;

// Instantiate PDController:
PDController controller(
  FRAME_WIDTH/2.0f,       // frame_cx
  FRAME_HEIGHT/2.0f,      // frame_cy
  Kp_x, Kd_x,             // Kp_x, Kd_x
  Kp_y, Kd_y,             // Kp_y, Kd_y
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
  base.write(90); // 90 is center, 0 is to the left, 180 is to the right
  servo_elbow.write(0); // 0 is straight ahead, 90 is up, 180 is all the way back        
  shoulder.write(45); // 0 all the way back, 45 a good neutral, 90 is straight up

  // record the default FK so we can hold x,z constant
  homePos = ik.getPositions();
  fixedX  = homePos.x;     // unused by your pan servo
  fixedZ  = homePos.z;     // keep depth constant
}

void loop() {
  // We expect 4 bytes from Serial per frame (face_x, face_y as two‐byte integers):
  if (Serial.available() < 4) return;
  // Read face_x (low‐byte, high‐byte)
  uint16_t face_x = Serial.read() | (Serial.read() << 8);
  // Read face_y (low‐byte, high‐byte)
  uint16_t face_y = Serial.read() | (Serial.read() << 8);

  // Run PD update. Returns { x = pan, y = tilt }
  auto cmd = controller.update((float)face_x, (float)face_y);
  int send_x = cmd.x;
  float send_y = cmd.y;
  
  base.write(send_x);

  // 2) VERTICAL plane → IK
  float targetY = homePos.y + cmd.y;

  // solve IK for (x= fixedX, y= targetY, z= fixedZ)
  ik.moveToPosition(fixedX, targetY, fixedZ);

  // 3) pull out the two joint angles
  Angle ang = ik.getAngles();
  // assume ang.a2 = shoulder angle, ang.a3 = elbow angle
  int shoulder_deg = (int)roundf(ang.theta2);
  int elbow_deg  = (int)roundf(ang.theta3);

  shoulder_deg = constrain(shoulder_deg, SHOULDER_MIN, SHOULDER_MAX);
  elbow_deg    = constrain(elbow_deg,    ELBOW_MIN,    ELBOW_MAX);

  shoulder.write(shoulder_deg);
  servo_elbow.write(elbow_deg);

  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 500) {
      Serial.printf("Face Y: %d, Δcm: %.2f, Shoulder: %d, Elbow: %d\n",
                    face_y, cmd.y, shoulder_deg, elbow_deg);
      lastPrint = millis();
  }
} 