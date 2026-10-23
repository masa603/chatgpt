#include <Arduino.h>

// Scanner button wiring: D2 = calibration, D3 = color reading.
// INPUT_PULLUP expects the switch to connect its pin to GND when pressed.
const uint8_t CALIB_BUTTON_PIN = 2;
const uint8_t READ_BUTTON_PIN = 3;

void setup() {
  Serial.begin(9600);
  pinMode(CALIB_BUTTON_PIN, INPUT_PULLUP);
  pinMode(READ_BUTTON_PIN, INPUT_PULLUP);

  delay(1000);
  Serial.println("BUTTON CHECK: D2=calibration, D3=reading");
  Serial.println("Expected: released=1, pressed=0. Hold each button for 2 seconds.");
  Serial.println("No image size or calibration input is required.");
}

void loop() {
  Serial.print("D2=");
  Serial.print(digitalRead(CALIB_BUTTON_PIN));
  Serial.print("  D3=");
  Serial.println(digitalRead(READ_BUTTON_PIN));
  delay(200);
}
