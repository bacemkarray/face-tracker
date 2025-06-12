#ifndef INVERSE_K_H
#define INVERSE_K_H

#include <Arduino.h>
#include <ESP32Servo.h>

/**
 * @brief 2-link planar inverse-kinematics controller with built-in servo mapping.
 * 
 * This class solves for shoulder/elbow angles in the X–Z plane and maps them
 * through your user-defined offsets into 0…180° servo positions (including base yaw).
 */
class InverseK {
    private:
        // link geometry
        const float L1, L2;

        // servo pins & offsets
        const int pinShoulder, pinElbow;
        const int shoulderOffset, elbowOffset;

        // Servo instances
        Servo shoulderServo, elbowServo;

        /** Convert logical angle (deg) + offset → 0…180 servo write */
        int toServoDeg(float logical_deg, int offset_deg) const;

    public:
    /**
     * @param link1_cm         Length of first link (shoulder→elbow) in cm
     * @param link2_cm         Length of second link (elbow→end-effector) in cm
     * @param pinShoulder      GPIO for shoulder (pitch) servo
     * @param pinElbow         GPIO for elbow (bend) servo
     * @param offsetShoulder_deg  same for shoulder
     * @param offsetElbow_deg     same for elbow
     */
    InverseK(float link1_cm,
             float link2_cm,
             int   pinShoulder,
             int   pinElbow,
             int   offsetShoulder_deg,
             int   offsetElbow_deg);

    // Attach all three servos. Call once in setup().
    void begin();

    /**
     * @brief Centers all joints to their “logical 0°” pose (vertical up).
     * @param shoulder_deg  logical shoulder pitch (default 90)
     * @param elbow_deg     logical elbow bend (default 90)
     */
    void home(int shoulder_deg = 90,
              int elbow_deg = 90);

    /**
     * @brief Solve IK for a target in the X–Z plane and immediately move servos.
     * @param x_cm    target X (forward) in cm
     * @param z_cm    target Z (up) in cm
     * @return true if target was within reach, false if clamped to limit
     */
    bool moveTo(float x_cm, float z_cm);

    /** Check reachability and return raw joint angles (radians) */
    bool solvePlanar(float x_cm, float z_cm,
                     float &outTheta1_rad,
                     float &outTheta2_rad) const;
};

#endif // INVERSE_K_H
