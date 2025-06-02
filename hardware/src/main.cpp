#include <Arduino.h>
#include <Servo.h>

Servo servo_x;
Servo servo_elbow;
Servo servo_arm;

void setup() {
  Serial.begin(115200);
  servo_x.attach(9);       
  servo_x.write(90);
  
  servo_arm.attach(11);
  servo_arm.write(0);

  servo_elbow.attach(10);
  servo_elbow.write(60);
}

void loop() {
  if (Serial.available() >= 4) { // Wait until a 4-byte packet is available
    uint16_t angleX = Serial.read(); // Read LSB of angleX
    angleX |= Serial.read() << 8; // Read MSB of angleX
    uint16_t angleY = Serial.read();
    angleY |= Serial.read() << 8;
    
    servo_x.write(angleX);
    // servo_elbow.write(angleY);
    servo_arm.write(angleY);
  }
  // if (Serial.available() >= 2) {
  //   int angleX = Serial.read();
  //   int angleY = Serial.read();
    
  //   servo_x.write(angleX);
  //   // servo_elbow.write(angleY);
  //   servo_arm.write(angleY);
    
  // }
}
