"""Build a one-file collector using the unchanged V7 sensor/normalization code."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_today.verify_fresh_scanner import function_body

source = (ROOT / 'distributed_training/color_retrain_v7/color_retrain_v7.ino').read_text()
destination = ROOT / 'distributed_training/patch_learning_capture'
destination.mkdir(exist_ok=True)
prefix = '''#include <Arduino.h>

// Additional labelled measurements only. This sketch does not run or train a model.
#define SAMPLE_COUNT 10
#define RGB_FLASH_DELAY 10
#define SENSOR_PIN A0
#define LED_R_PIN 8
#define LED_G_PIN 6
#define LED_B_PIN 7
constexpr int REPEATS = 3;
float rawRGB[3] = {0,0,0};
float blackRGB[3] = {0,0,0};
float whiteRGB[3] = {0,0,0};
int calibrationStep = 0;
unsigned long calibrationNumber = 0;
unsigned long batchNumber = 0;

'''
functions = '\n\n'.join(signature + function_body(source, name) for name, signature in [
    ('ledOff', 'void ledOff() '),
    ('readAverage', 'float readAverage() '),
    ('readRGB', 'void readRGB() '),
    ('normalizeInput', 'float normalizeInput(float value, float blackValue, float whiteValue) '),
])
interface = r'''

void printCalibration(const char *name, const float *values) {
  Serial.print(name);
  for (int c=0;c<3;c++) { Serial.print(","); Serial.print(values[c],4); }
  Serial.println();
}

void help() {
  Serial.println("黒 b → 白 w で校正します。");
  Serial.println("次に 紫 p → 黒 k → 灰 g → 白 v。各文字を1回送信すると3回自動で読みます。");
  Serial.println("読取のSTARTからENDまでは同じ色を当ててください。");
  Serial.println("PATCH LEARNING START〜ENDの4組を保存してください。最初からは n。");
}

void measurePatch(const char *id, const char *label) {
  if (calibrationStep!=2) { Serial.println("先に黒 b → 白 w で校正してください。"); return; }
  ++batchNumber;
  Serial.println("========== PATCH LEARNING START ==========");
  Serial.println("FIRMWARE=PATCH_LEARNING_CAPTURE_V1");
  Serial.println("INPUT_SOURCE=NEW_PHYSICAL_SENSOR_READINGS");
  Serial.println("INTENDED_ROLE=ADDITIONAL_TRAINING_DATA; NOT_A_FULL_CHART_MSE");
  Serial.println("LABEL_SOURCE=USER_COMMAND; NOT_INFERRED_FROM_INPUT");
  Serial.println("REFERENCE_SOURCE=DISTRIBUTED_NOTEBOOK_CMYK_DISPLAY_RECOVERED");
  Serial.println("REFERENCE_FILE_SHA256=4f58e236c2e114ed64e6e4128b94265ea66de1ee23413a7ef8b6a3c999f2ff14");
  Serial.print("CALIBRATION="); Serial.println(calibrationNumber);
  Serial.print("BATCH="); Serial.println(batchNumber);
  Serial.print("PATCH_ID="); Serial.println(id);
  Serial.print("PATCH_LABEL="); Serial.println(label);
  Serial.println("REPEATS=3; RAPID_WITHIN_ONE_PLACEMENT; NOT_THREE_INDEPENDENT_CAPTURES");
  Serial.println("SAMPLE_COUNT=10; RGB_FLASH_DELAY_MS=10; LED_DRIVE=ANALOG_WRITE_255_0");
  printCalibration("# black_rgb",blackRGB);
  printCalibration("# white_rgb",whiteRGB);
  Serial.println("repeat,sample_completed_ms,raw_r,raw_g,raw_b,input_r,input_g,input_b,unclipped_r,unclipped_g,unclipped_b");
  for (int repeat=1;repeat<=REPEATS;repeat++) {
    readRGB();
    unsigned long completed=millis();
    Serial.print(repeat); Serial.print(","); Serial.print(completed);
    for (int c=0;c<3;c++) { Serial.print(","); Serial.print(rawRGB[c],4); }
    for (int c=0;c<3;c++) { Serial.print(","); Serial.print(normalizeInput(rawRGB[c],blackRGB[c],whiteRGB[c]),8); }
    for (int c=0;c<3;c++) { Serial.print(","); Serial.print((rawRGB[c]-blackRGB[c])/(whiteRGB[c]-blackRGB[c]),8); }
    Serial.println();
    if (repeat<REPEATS) delay(100);
  }
  Serial.println("=========== PATCH LEARNING END ===========");
  Serial.println("次の色を当てて p / k / g / v を1回送信してください。");
}

void handleCommand(char command) {
  if (command>='A' && command<='Z') command=command-'A'+'a';
  if (command=='\r' || command=='\n' || command==' ' || command=='\t') return;
  if (command=='n') { calibrationStep=0; help(); return; }
  if (command=='b') {
    readRGB();
    for (int c=0;c<3;c++) blackRGB[c]=rawRGB[c];
    calibrationStep=1;
    Serial.println("黒を登録しました。白を当てて w を送信してください。");
    printCalibration("# black_rgb",blackRGB);
    return;
  }
  if (command=='w') {
    if (calibrationStep!=1) { Serial.println("白の登録は黒 b の後です。読取の白は v です。"); return; }
    readRGB();
    bool valid=true;
    for (int c=0;c<3;c++) { whiteRGB[c]=rawRGB[c]; valid=valid && whiteRGB[c]>blackRGB[c]; }
    if (!valid) { calibrationStep=0; Serial.println("校正失敗。黒 b からやり直してください。"); return; }
    calibrationStep=2; ++calibrationNumber;
    printCalibration("# white_rgb",whiteRGB);
    Serial.println("校正完了。紫 p → 黒 k → 灰 g → 白 v を各1回送信してください。");
    return;
  }
  if (command=='p') { measurePatch("r2c3","PURPLE"); return; }
  if (command=='k') { measurePatch("r3c1","BLACK"); return; }
  if (command=='g') { measurePatch("r3c2","GRAY"); return; }
  if (command=='v') { measurePatch("r3c3","WHITE"); return; }
  help();
}

void setup() {
  Serial.begin(9600);
  pinMode(LED_R_PIN,OUTPUT); pinMode(LED_G_PIN,OUTPUT); pinMode(LED_B_PIN,OUTPUT);
  ledOff(); delay(1000);
  Serial.println("FIRMWARE=PATCH_LEARNING_CAPTURE_V1");
  Serial.println("学習用データの収集です。新しい学習済みモデルやMSEの結果は出しません。");
  help();
}

void loop() {
  if (Serial.available()) handleCommand(static_cast<char>(Serial.read()));
  delay(1);
}
'''
(destination / 'patch_learning_capture.ino').write_text(prefix+functions+interface)
print('Generated one-file patch collector; original V7 measurement and clipped NN input are unchanged.')
