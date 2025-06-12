#include "InverseK.h"
#include <math.h>

InverseK::InverseK(float link1_cm, float link2_cm,
                   int pinShoulder, int pinElbow,
                   int   offsetShoulder_deg,
                   int   offsetElbow_deg)
  : L1(link1_cm)
  , L2(link2_cm)
  , pinShoulder(pinShoulder)
  , pinElbow(pinElbow)
  , shoulderOffset(offsetShoulder_deg)
  , elbowOffset(offsetElbow_deg)
{}

void InverseK::begin() {
  shoulderServo.attach(pinShoulder);
  elbowServo.attach(pinElbow);
}

void InverseK::home(int shoulder_deg,
                    int elbow_deg) {
  shoulderServo.write( toServoDeg(shoulder_deg, shoulderOffset) );
  elbowServo.write( toServoDeg(elbow_deg, elbowOffset) );
}

bool InverseK::solvePlanar(float x_cm,
                           float z_cm,
                           float &outTheta1_rad,
                           float &outTheta2_rad) const
{
  float r2 = x_cm*x_cm + z_cm*z_cm;
  float r  = sqrt(r2);

  // law‐of‐cosines for theta2
  float c2 = (r2 - L1*L1 - L2*L2) / (2 * L1 * L2);
  c2 = constrain(c2, -1.0f, +1.0f);

  float s2 = sqrt(1 - c2*c2);
  outTheta2_rad = atan2(s2, c2);  // “elbow-up” solution

  // theta1 from geometry
  float k1 = L1 + L2 * c2;
  float k2 = L2 * s2;
  outTheta1_rad = atan2(z_cm, x_cm) - atan2(k2, k1);

  // reachable if distance ∈ [|L1−L2|, L1+L2]
  return (r <= (L1+L2) && r >= fabs(L1 - L2));
}

bool InverseK::moveTo(float x_cm, float z_cm) {
  float th1_rad, th2_rad;
  bool reachable = solvePlanar(x_cm, z_cm, th1_rad, th2_rad);

  // convert to degrees
  float th1_deg = th1_rad * 180.0f / M_PI;
  float th2_deg = th2_rad * 180.0f / M_PI;

  // map logical → servo positions
  // logical 0° → servo at (90 + offset)
  int shoulderPos = toServoDeg(th1_deg, shoulderOffset);
  int elbowPos    = toServoDeg(th2_deg,    elbowOffset);

  shoulderServo.write(shoulderPos);
  elbowServo.write(elbowPos);

  return reachable;
}

int InverseK::toServoDeg(float logical_deg, int offset_deg) const {
  // logical_deg + 90 sets “logical 0°” at servo 90,
  // then we add offset to shift your hardware zero.
  int raw = (int)round(logical_deg + 90.0f + offset_deg);
  return constrain(raw, 0, 180);
}
