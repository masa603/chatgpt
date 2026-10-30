#include <Arduino.h>

constexpr uint8_t SENSOR_PIN = A0;
constexpr uint8_t LED_PINS[3] = {8, 6, 7};  // R, G, B: same wiring as scanner
constexpr uint8_t SAMPLE_COUNT = 10;
constexpr uint16_t SETTLE_DELAYS[2] = {10, 200};
constexpr uint8_t REPEATS = 5;
constexpr uint16_t INTER_READ_PAUSE_MS = 200;
uint16_t captureNumber = 0;

void ledOff() {
  for (uint8_t color = 0; color < 3; color++) digitalWrite(LED_PINS[color], LOW);
}

float readAverage() {
  long sum = 0;
  for (uint8_t sample = 0; sample < SAMPLE_COUNT; sample++) {
    sum += analogRead(SENSOR_PIN);
    delay(2);
  }
  return static_cast<float>(sum) / SAMPLE_COUNT;
}

void readRGB(uint16_t settleMs, float output[3]) {
  for (uint8_t color = 0; color < 3; color++) {
    ledOff();
    // Only full on/off is required; no PWM timer is used in this diagnostic.
    digitalWrite(LED_PINS[color], HIGH);
    delay(settleMs);
    output[color] = readAverage();
    digitalWrite(LED_PINS[color], LOW);
  }
}

void captureRaw(const char *patch) {
  captureNumber++;
  uint32_t start = static_cast<uint32_t>(millis());
  Serial.println("========== RAW CHECK START ==========");
  Serial.print("CAPTURE="); Serial.println(captureNumber);
  Serial.print("PATCH="); Serial.println(patch);
  Serial.println("LED_DRIVE=DIGITAL_FULL_ON_OFF");
  Serial.print("SAMPLE_COUNT="); Serial.println(SAMPLE_COUNT);
  Serial.print("INTER_READ_PAUSE_MS="); Serial.println(INTER_READ_PAUSE_MS);
  Serial.println("Keep the paper still until RAW CHECK END.");
  Serial.println("delay_ms,repeat,elapsed_ms,raw_r,raw_g,raw_b");
  for (uint8_t condition = 0; condition < 2; condition++) {
    for (uint8_t repeat = 0; repeat < REPEATS; repeat++) {
      float rgb[3];
      readRGB(SETTLE_DELAYS[condition], rgb);
      Serial.print(SETTLE_DELAYS[condition]); Serial.print(",");
      Serial.print(repeat + 1); Serial.print(",");
      Serial.print(static_cast<uint32_t>(millis()) - start);
      for (uint8_t channel = 0; channel < 3; channel++) {
        Serial.print(","); Serial.print(rgb[channel], 2);
      }
      Serial.println();
      delay(INTER_READ_PAUSE_MS);
    }
  }
  ledOff();
  Serial.println("=========== RAW CHECK END ===========");
  Serial.println("Next: move paper, then send b (black) or w (white).");
}

void setup() {
  Serial.begin(9600);
  for (uint8_t color = 0; color < 3; color++) pinMode(LED_PINS[color], OUTPUT);
  ledOff();
  delay(1000);
  Serial.println("Raw sensor stability check: black b -> white w -> same black b.");
  Serial.println("Hold each patch still until RAW CHECK END (about 6 seconds).");
}

void loop() {
  if (Serial.available() > 0) {
    int command = Serial.read();
    if (command == 'b' || command == 'B') captureRaw("BLACK");
    else if (command == 'w' || command == 'W') captureRaw("WHITE");
    else if (command != '\r' && command != '\n' && command != ' ' && command != '\t') {
      Serial.println("Send b for black or w for white.");
    }
  }
}
