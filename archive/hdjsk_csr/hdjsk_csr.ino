#include <Arduino.h>
#include <cmath>

#ifndef ENABLE_CSR_SELF_TEST
#define ENABLE_CSR_SELF_TEST 1
#endif
#include "model_parameters_csr.h"

// =====================================================
// ニューラルネット設定
// =====================================================

#define INPUT_SIZE 3
#define HIDDEN_SIZE1 CSR_HIDDEN_SIZE
#define HIDDEN_SIZE2 CSR_HIDDEN_SIZE
#define OUTPUT_SIZE 3

// =====================================================
// スキャナ設定
// =====================================================

#define SAMPLE_COUNT 10
#define RGB_FLASH_DELAY 10
#define DEBOUNCE_MS 40

#define MAX_IMAGE_SIZE 8

// 0：シリアルモニタで操作、1：D2/D3ボタンも使用
#define USE_PHYSICAL_BUTTONS 0

#define SENSOR_PIN A0

#define LED_R_PIN 8
#define LED_G_PIN 6
#define LED_B_PIN 7

// D2：黒・白キャリブレーション
#define BUTTON_CALIB_PIN 2

// D3：色情報読み取り
#define BUTTON_READ_PIN 3


// =====================================================
// ボタン
// =====================================================

struct DebouncedButton {
  uint8_t pin;
  int rawState;
  int stableState;
  unsigned long lastChangeTime;
};

DebouncedButton calibButton;
DebouncedButton readButton;


// =====================================================
// RGB
// =====================================================

float rawRGB[3] = {0.0, 0.0, 0.0};

float blackRGB[3] = {0.0, 0.0, 0.0};
float whiteRGB[3] = {1023.0, 1023.0, 1023.0};

// NNに入力する0.0～1.0の値
float RGBInput[3] = {0.0, 0.0, 0.0};

// NNから出力される値
float RGBOutput[3] = {0.0, 0.0, 0.0};

// 0：黒待ち
// 1：白待ち
// 2：キャリブレーション完了
int calibrationStep = 0;


// =====================================================
// 画像
// =====================================================

int imageSize = 0;

uint8_t imageData[MAX_IMAGE_SIZE][MAX_IMAGE_SIZE][3];

// 全色の正規化済み入力を保存し、読取完了後にまとめて表示する
float inputHistory[MAX_IMAGE_SIZE][MAX_IMAGE_SIZE][3];

int currentRow = 0;
int currentColumn = 0;

bool imageFinished = false;


// =====================================================
// ReLU
// =====================================================

float relu(float x) {
  return (x > 0) ? x : 0;
}


// =====================================================
// ニューラルネット推論
// =====================================================

void csrLayer(
  const float *input, const float *values, const uint8_t *columns,
  const uint16_t *rowPointers, const float *bias, uint16_t rows,
  float *output, bool useRelu
) {
  for (uint16_t row = 0; row < rows; row++) {
    float sum = bias[row];
    for (uint16_t p = rowPointers[row]; p < rowPointers[row + 1]; p++) {
      sum += values[p] * input[columns[p]];
    }
    output[row] = useRelu ? relu(sum) : sum;
  }
}

void predictLayers(const float *input, float *layer1, float *layer2, float *output) {
  csrLayer(input, weight_1_values, weight_1_col_idx, weight_1_row_ptr,
           bias_1, HIDDEN_SIZE1, layer1, true);
  csrLayer(layer1, weight_2_values, weight_2_col_idx, weight_2_row_ptr,
           bias_2, HIDDEN_SIZE2, layer2, true);
  csrLayer(layer2, weight_3_values, weight_3_col_idx, weight_3_row_ptr,
           bias_3, OUTPUT_SIZE, output, false);
}

void predict(float input[INPUT_SIZE], float output[OUTPUT_SIZE]) {
  float layer1[HIDDEN_SIZE1] = {0};
  float layer2[HIDDEN_SIZE2] = {0};
  predictLayers(input, layer1, layer2, output);
}

size_t csrModelBytes() {
  return sizeof(weight_1_values) + sizeof(weight_1_col_idx) + sizeof(weight_1_row_ptr) + sizeof(bias_1)
       + sizeof(weight_2_values) + sizeof(weight_2_col_idx) + sizeof(weight_2_row_ptr) + sizeof(bias_2)
       + sizeof(weight_3_values) + sizeof(weight_3_col_idx) + sizeof(weight_3_row_ptr) + sizeof(bias_3);
}

bool validCSRLayer(
  const float *values, const uint8_t *columns, const uint16_t *rowPointers,
  uint16_t rows, uint16_t inputColumns, size_t nonzeros
) {
  if (rowPointers[0] != 0 || rowPointers[rows] != nonzeros) {
    return false;
  }
  for (uint16_t row = 0; row < rows; row++) {
    if (rowPointers[row] > rowPointers[row + 1] || rowPointers[row + 1] > nonzeros) {
      return false;
    }
    for (uint16_t p = rowPointers[row]; p < rowPointers[row + 1]; p++) {
      if (columns[p] >= inputColumns || !std::isfinite(values[p]) || values[p] == 0.0f) {
        return false;
      }
      if (p > rowPointers[row] && columns[p] <= columns[p - 1]) {
        return false;
      }
    }
  }
  return true;
}

bool validCSRModel() {
  return validCSRLayer(weight_1_values, weight_1_col_idx, weight_1_row_ptr,
                       HIDDEN_SIZE1, INPUT_SIZE, sizeof(weight_1_values) / sizeof(float))
      && validCSRLayer(weight_2_values, weight_2_col_idx, weight_2_row_ptr,
                       HIDDEN_SIZE2, HIDDEN_SIZE1, sizeof(weight_2_values) / sizeof(float))
      && validCSRLayer(weight_3_values, weight_3_col_idx, weight_3_row_ptr,
                       OUTPUT_SIZE, HIDDEN_SIZE2, sizeof(weight_3_values) / sizeof(float));
}

#if ENABLE_CSR_SELF_TEST
bool checkLayerOutput(const float *actual, const float *expected, int count, float &maxError) {
  bool passed = true;
  for (int i = 0; i < count; i++) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
      passed = false;
      continue;
    }
    float error = std::fabs(actual[i] - expected[i]);
    if (error > maxError) {
      maxError = error;
    }
    // FP32 tolerance: 1e-5 absolute plus 1e-5 relative, before RGB clipping.
    float tolerance = 1.0e-5f + 1.0e-5f * std::fabs(expected[i]);
    if (error > tolerance) {
      passed = false;
    }
  }
  return passed;
}

bool runCSRSelfTest(float maxErrors[3]) {
  bool passed = true;
  maxErrors[0] = maxErrors[1] = maxErrors[2] = 0.0f;
  for (int sample = 0; sample < CSR_SELF_TEST_CASES; sample++) {
    float layer1[HIDDEN_SIZE1] = {0};
    float layer2[HIDDEN_SIZE2] = {0};
    float output[OUTPUT_SIZE] = {0};
    predictLayers(CSR_SELF_TEST_INPUTS[sample], layer1, layer2, output);
    if (!checkLayerOutput(layer1, CSR_SELF_TEST_LAYER1[sample], HIDDEN_SIZE1, maxErrors[0])) passed = false;
    if (!checkLayerOutput(layer2, CSR_SELF_TEST_LAYER2[sample], HIDDEN_SIZE2, maxErrors[1])) passed = false;
    if (!checkLayerOutput(output, CSR_SELF_TEST_OUTPUT[sample], OUTPUT_SIZE, maxErrors[2])) passed = false;
  }
  return passed;
}
#endif

bool printCSRDiagnostics() {
  Serial.println("========== CSR CHECK START ==========");
  Serial.print("CSR MODEL: ");
  Serial.println(CSR_PROVISIONAL ? "PROVISIONAL" : "SELECTED");
  Serial.print("H=");
  Serial.println(CSR_HIDDEN_SIZE);
  Serial.print("THRESHOLD=");
  Serial.println(CSR_THRESHOLD_TEXT);
  Serial.print("MODEL BYTES: Dense=");
  Serial.print(CSR_DENSE_REFERENCE_BYTES);
  Serial.print(" CSR=");
  Serial.println(csrModelBytes());
  Serial.print("MODEL NNZ=");
  Serial.println(CSR_NONZERO_COUNT);
  Serial.print("MODEL SOURCE SHA256=");
  Serial.println(CSR_SOURCE_MODEL_SHA256);
  if (csrModelBytes() != CSR_EXPECTED_MODEL_BYTES || !validCSRModel()) {
    Serial.println("CSR INDEX CHECK: FAIL");
    return false;
  }
  Serial.println("CSR INDEX CHECK: PASS");
#if ENABLE_CSR_SELF_TEST
  float maxErrors[3] = {0};
  bool passed = runCSRSelfTest(maxErrors);
  Serial.print("CSR SELF TEST: ");
  Serial.print(passed ? "PASS" : "FAIL");
  Serial.print(" (");
  Serial.print(CSR_SELF_TEST_CASES);
  Serial.println(" cases)");
  Serial.print("MAX ABS ERROR: L1=");
  Serial.print(maxErrors[0], 8);
  Serial.print(" L2=");
  Serial.print(maxErrors[1], 8);
  Serial.print(" OUT=");
  Serial.println(maxErrors[2], 8);
  Serial.print("SELF TEST REFERENCE BYTES=");
  Serial.println(sizeof(CSR_SELF_TEST_INPUTS) + sizeof(CSR_SELF_TEST_LAYER1)
                 + sizeof(CSR_SELF_TEST_LAYER2) + sizeof(CSR_SELF_TEST_OUTPUT));
  Serial.println("=========== CSR CHECK END ===========");
  return passed;
#else
  Serial.println("CSR SELF TEST: DISABLED");
  Serial.println("=========== CSR CHECK END ===========");
  return true;
#endif
}


// =====================================================
// ボタン初期化
// =====================================================

void initButton(
  DebouncedButton &button,
  uint8_t pin
) {

  button.pin = pin;

  pinMode(pin, INPUT_PULLUP);

  int state = digitalRead(pin);

  button.rawState = state;
  button.stableState = state;
  button.lastChangeTime = millis();
}


// =====================================================
// ボタンを押した瞬間だけtrue
// =====================================================

bool buttonPressed(DebouncedButton &button) {

  int currentState = digitalRead(button.pin);
  unsigned long now = millis();


  if (currentState != button.rawState) {

    button.rawState = currentState;
    button.lastChangeTime = now;
  }


  if (
    now - button.lastChangeTime
    >= DEBOUNCE_MS
  ) {

    if (
      button.stableState
      != button.rawState
    ) {

      button.stableState =
        button.rawState;


      // HIGH → LOW = 押された瞬間
      if (button.stableState == LOW) {

        return true;
      }
    }
  }


  return false;
}


// =====================================================
// LED OFF
// =====================================================

void ledOff() {

  analogWrite(LED_R_PIN, 0);
  analogWrite(LED_G_PIN, 0);
  analogWrite(LED_B_PIN, 0);
}


// =====================================================
// センサを10回読んで平均
// =====================================================

float readAverage() {

  long sum = 0;

  for (int i = 0; i < SAMPLE_COUNT; i++) {

    sum += analogRead(SENSOR_PIN);

    delay(2);
  }

  return (float)sum / SAMPLE_COUNT;
}


// =====================================================
// RGB読み取り
// =====================================================

void readRGB() {

  // -------------------------
  // RED
  // -------------------------

  analogWrite(LED_R_PIN, 255);
  analogWrite(LED_G_PIN, 0);
  analogWrite(LED_B_PIN, 0);

  delay(RGB_FLASH_DELAY);

  rawRGB[0] = readAverage();


  // -------------------------
  // GREEN
  // -------------------------

  analogWrite(LED_R_PIN, 0);
  analogWrite(LED_G_PIN, 255);
  analogWrite(LED_B_PIN, 0);

  delay(RGB_FLASH_DELAY);

  rawRGB[1] = readAverage();


  // -------------------------
  // BLUE
  // -------------------------

  analogWrite(LED_R_PIN, 0);
  analogWrite(LED_G_PIN, 0);
  analogWrite(LED_B_PIN, 255);

  delay(RGB_FLASH_DELAY);

  rawRGB[2] = readAverage();


  ledOff();
}


// =====================================================
// 黒～白を0.0～1.0へ正規化
// =====================================================

float normalizeInput(
  float value,
  float blackValue,
  float whiteValue
) {

  float range =
    whiteValue - blackValue;


  if (range <= 0.0) {
    return 0.0;
  }


  float result =
    (value - blackValue)
    / range;


  if (result < 0.0) {
    result = 0.0;
  }

  if (result > 1.0) {
    result = 1.0;
  }


  return result;
}


// =====================================================
// NN出力を0～255へ変換
// =====================================================

int outputTo255(float value) {

  // 出力層にはReLUを使っていないため
  // 最後に0～1へクリップする

  if (value < 0.0) {
    value = 0.0;
  }

  if (value > 1.0) {
    value = 1.0;
  }


  return (int)(
    value * 255.0
    + 0.5
  );
}


// =====================================================
// 黒・白キャリブレーション
// =====================================================

void handleCalibration() {

  // ===================================================
  // 黒を登録
  // ===================================================

  if (calibrationStep == 0) {

    readRGB();


    blackRGB[0] = rawRGB[0];
    blackRGB[1] = rawRGB[1];
    blackRGB[2] = rawRGB[2];


    Serial.println();
    Serial.println("============================");
    Serial.println("黒を登録しました");
    Serial.println("============================");


    Serial.print("Black R = ");
    Serial.println(blackRGB[0]);

    Serial.print("Black G = ");
    Serial.println(blackRGB[1]);

    Serial.print("Black B = ");
    Serial.println(blackRGB[2]);


    Serial.println();

    Serial.println(
      "白を当てて w を送信してください"
    );


    calibrationStep = 1;

    return;
  }


  // ===================================================
  // 白を登録
  // ===================================================

  if (calibrationStep == 1) {

    readRGB();


    whiteRGB[0] = rawRGB[0];
    whiteRGB[1] = rawRGB[1];
    whiteRGB[2] = rawRGB[2];


    Serial.println();
    Serial.println("============================");
    Serial.println("白を登録しました");
    Serial.println("============================");


    Serial.print("White R = ");
    Serial.println(whiteRGB[0]);

    Serial.print("White G = ");
    Serial.println(whiteRGB[1]);

    Serial.print("White B = ");
    Serial.println(whiteRGB[2]);


    // 白が黒より大きいか確認
    if (
      whiteRGB[0] <= blackRGB[0]
      ||
      whiteRGB[1] <= blackRGB[1]
      ||
      whiteRGB[2] <= blackRGB[2]
    ) {

      Serial.println();

      Serial.println(
        "キャリブレーション失敗"
      );

      Serial.println(
        "黒からやり直してください"
      );


      calibrationStep = 0;

      return;
    }


    calibrationStep = 2;


    Serial.println();
    Serial.println("============================");
    Serial.println("キャリブレーション完了");
    Serial.println("============================");

    Serial.println();

    Serial.println(
      "色を当てて r を送信すると読み取ります"
    );

    Serial.println();

    return;
  }


  Serial.println(
    "キャリブレーション済みです。r を送信すると読み取ります。"
  );
}


// =====================================================
// PPM出力
// =====================================================

void outputPPM() {

  Serial.println();

  Serial.println("==============================");
  Serial.println("画像読み取り完了");
  Serial.println("==============================");

  Serial.println();

  Serial.println("---------- PPM START ----------");

  Serial.println("P3");

  Serial.print(imageSize);
  Serial.print(" ");
  Serial.println(imageSize);

  Serial.println("255");


  for (int row = 0; row < imageSize; row++) {

    for (int col = 0; col < imageSize; col++) {

      Serial.print(
        imageData[row][col][0]
      );

      Serial.print(" ");

      Serial.print(
        imageData[row][col][1]
      );

      Serial.print(" ");

      Serial.print(
        imageData[row][col][2]
      );

      Serial.print(" ");
    }

    Serial.println();
  }


  Serial.println("----------- PPM END -----------");
}


// =====================================================
// 読取完了後のNN入力一覧（PPMは補正後、この一覧は補正前）
// =====================================================

void outputNNInputs() {
  Serial.println();
  Serial.println("---------- NN INPUT START ----------");
  Serial.print("# size,");
  Serial.print(imageSize);
  Serial.print(",");
  Serial.println(imageSize);

  Serial.print("# black_rgb");
  for (int channel = 0; channel < 3; channel++) {
    Serial.print(",");
    Serial.print(blackRGB[channel], 4);
  }
  Serial.println();

  Serial.print("# white_rgb");
  for (int channel = 0; channel < 3; channel++) {
    Serial.print(",");
    Serial.print(whiteRGB[channel], 4);
  }
  Serial.println();
  Serial.println("color_id,input_r,input_g,input_b");

  for (int row = 0; row < imageSize; row++) {
    for (int col = 0; col < imageSize; col++) {
      // color_idは1始まり。例：r1c1は左上、r8c8は右下。
      Serial.print("r");
      Serial.print(row + 1);
      Serial.print("c");
      Serial.print(col + 1);
      for (int channel = 0; channel < 3; channel++) {
        Serial.print(",");
        Serial.print(inputHistory[row][col][channel], 8);
      }
      Serial.println();
    }
  }
  Serial.println("----------- NN INPUT END -----------");
}


// =====================================================
// 色読み取り + NN補正
// =====================================================

void handleRead() {

  // キャリブレーション前
  if (calibrationStep != 2) {

    Serial.println(
      "先に黒を当てて b、白を当てて w を順に送信してください"
    );

    return;
  }


  // 読み取り済み
  if (imageFinished) {

    Serial.println(
      "読み取りはすでに終了しています"
    );

    return;
  }


  // ===================================================
  // センサ読み取り
  // ===================================================

  readRGB();


  // ===================================================
  // NN入力用に0～1へ正規化
  // ===================================================

  RGBInput[0] =
    normalizeInput(
      rawRGB[0],
      blackRGB[0],
      whiteRGB[0]
    );


  RGBInput[1] =
    normalizeInput(
      rawRGB[1],
      blackRGB[1],
      whiteRGB[1]
    );


  RGBInput[2] =
    normalizeInput(
      rawRGB[2],
      blackRGB[2],
      whiteRGB[2]
    );


  // ===================================================
  // ニューラルネットで補正
  // ===================================================

  predict(
    RGBInput,
    RGBOutput
  );


  // ===================================================
  // 0～255へ変換
  // ===================================================

  int red =
    outputTo255(
      RGBOutput[0]
    );


  int green =
    outputTo255(
      RGBOutput[1]
    );


  int blue =
    outputTo255(
      RGBOutput[2]
    );


  // ===================================================
  // シリアル表示
  // ===================================================

  Serial.println();


  Serial.println("--- センサ値 ---");

  Serial.print("RAW R:");
  Serial.print(rawRGB[0]);

  Serial.print(" G:");
  Serial.print(rawRGB[1]);

  Serial.print(" B:");
  Serial.println(rawRGB[2]);


  Serial.println("--- NN入力 ---");

  Serial.print("R:");
  Serial.print(RGBInput[0], 4);

  Serial.print(" G:");
  Serial.print(RGBInput[1], 4);

  Serial.print(" B:");
  Serial.println(RGBInput[2], 4);


  Serial.println("--- NN補正後 ---");

  Serial.print("Red:");
  Serial.print(red);

  Serial.print(" Green:");
  Serial.print(green);

  Serial.print(" Blue:");
  Serial.println(blue);


  // ===================================================
  // 画像へ保存
  // ===================================================

  imageData[currentRow][currentColumn][0] = red;
  imageData[currentRow][currentColumn][1] = green;
  imageData[currentRow][currentColumn][2] = blue;

  for (int channel = 0; channel < 3; channel++) {
    inputHistory[currentRow][currentColumn][channel] = RGBInput[channel];
  }


  Serial.print(currentRow);
  Serial.print(" 行目 ");

  Serial.print(currentColumn);
  Serial.println(" 列目 を読み込みました");


  // ===================================================
  // 次の画素へ
  // ===================================================

  currentColumn++;


  if (currentColumn >= imageSize) {

    currentColumn = 0;

    currentRow++;
  }


  // ===================================================
  // 全画素終了
  // ===================================================

  if (currentRow >= imageSize) {

    imageFinished = true;

    outputPPM();
    outputNNInputs();
  }
}


// =====================================================
// シリアルモニタから操作
// =====================================================

void handleSerialCommand() {
  if (Serial.available() <= 0) {
    return;
  }

  int command = Serial.read();

  switch (command) {
    case 'b':
    case 'B':
      if (calibrationStep != 0) {
        Serial.println("黒は登録済みです。白は w、読取は r を送信してください。");
        return;
      }
      Serial.println("黒を測定します");
      handleCalibration();
      break;

    case 'w':
    case 'W':
      if (calibrationStep != 1) {
        Serial.println("白の登録は黒の登録後です。黒は b、読取は r を送信してください。");
        return;
      }
      Serial.println("白を測定します");
      handleCalibration();
      break;

    case 'r':
    case 'R':
      handleRead();
      break;

    case '\r':
    case '\n':
    case ' ':
    case '\t':
      break;

    default:
      Serial.println("半角で b（黒）、w（白）、r（読取）を送信してください。");
      break;
  }
}


// =====================================================
// 画像サイズ入力
// =====================================================

void inputImageSize() {

  Serial.println(
    "画像サイズを入力してください"
  );

  Serial.println(
    "3x3なら 3"
  );

  Serial.println(
    "6x6なら 6"
  );

  Serial.println(
    "8x8なら 8"
  );


  while (imageSize == 0) {

    if (Serial.available() > 0) {

      int value =
        Serial.parseInt();


      // 改行などを削除
      while (Serial.available() > 0) {
        Serial.read();
      }


      if (
        value >= 1
        &&
        value <= MAX_IMAGE_SIZE
      ) {

        imageSize = value;

      } else {

        Serial.print("1～");
        Serial.print(MAX_IMAGE_SIZE);
        Serial.println("を入力してください");
      }
    }


    delay(10);
  }


  Serial.println();

  Serial.print("画像サイズ：");

  Serial.print(imageSize);

  Serial.print(" x ");

  Serial.println(imageSize);
}


// =====================================================
// setup
// =====================================================

void setup() {

  Serial.begin(9600);

  delay(1000);


  // ===================================================
  // LED
  // ===================================================

  pinMode(
    LED_R_PIN,
    OUTPUT
  );

  pinMode(
    LED_G_PIN,
    OUTPUT
  );

  pinMode(
    LED_B_PIN,
    OUTPUT
  );


  ledOff();


  // ===================================================
  // ボタン
  // ===================================================

#if USE_PHYSICAL_BUTTONS
  initButton(
    calibButton,
    BUTTON_CALIB_PIN
  );


  initButton(
    readButton,
    BUTTON_READ_PIN
  );
#endif


  // ===================================================
  // 画像初期化
  // ===================================================

  for (
    int row = 0;
    row < MAX_IMAGE_SIZE;
    row++
  ) {

    for (
      int col = 0;
      col < MAX_IMAGE_SIZE;
      col++
    ) {

      imageData[row][col][0] = 0;
      imageData[row][col][1] = 0;
      imageData[row][col][2] = 0;
    }
  }


  if (!printCSRDiagnostics()) {
    Serial.println("CSR check failed. Save the log and check the sketch/header.");
    while (true) {
      delay(1000);
    }
  }

  inputImageSize();


  Serial.println();

  Serial.println("============================");
  Serial.println("枝刈り＋CSR版準備完了");
  Serial.println("============================");

  Serial.println();

  Serial.println("① 黒を当てて b を送信");
  Serial.println("② 白を当てて w を送信");
  Serial.println("③ 各色を当てて r を1回ずつ送信");
#if USE_PHYSICAL_BUTTONS
  Serial.println("D2：黒・白登録、D3：読取も使用できます");
#else
  Serial.println("D2/D3ボタンは使いません");
#endif

  Serial.println();
}


// =====================================================
// loop
// =====================================================

void loop() {

  handleSerialCommand();

#if USE_PHYSICAL_BUTTONS
  bool d2Pressed =
    buttonPressed(
      calibButton
    );


  bool d3Pressed =
    buttonPressed(
      readButton
    );


  if (d2Pressed) {

    handleCalibration();
  }


  if (d3Pressed) {

    handleRead();
  }
#endif


  delay(1);
}
