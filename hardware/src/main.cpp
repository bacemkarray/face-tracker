#include <Arduino.h>
#include <Servo.h>

Servo servoX;
Servo servoY;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);     // Match this with Python serial baudrate
  servoX.attach(9);       // Set your actual servo pin
  servoY.attach(10);      // Set your actual servo pin
  servoX.write(90);
  servoY.write(90);
}

void loop() {
  if (Serial.available() >= 2) {
    int angleX = Serial.read();
    int angleY = Serial.read();

    servoX.write(angleX);
    servoY.write(angleY);
  }
}