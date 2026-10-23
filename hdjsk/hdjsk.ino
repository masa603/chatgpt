#include <Arduino.h>
#include "model_parameters.h"

// =====================================================
// ニューラルネット設定
// =====================================================

#define INPUT_SIZE 3
#define HIDDEN_SIZE1 40
#define HIDDEN_SIZE2 40
#define OUTPUT_SIZE 3

// =====================================================
// スキャナ設定
// =====================================================

#define SAMPLE_COUNT 10
#define RGB_FLASH_DELAY 10
#define DEBOUNCE_MS 40

#define MAX_IMAGE_SIZE 6

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

void predict(float input[INPUT_SIZE], float output[OUTPUT_SIZE]) {

  float layer1[HIDDEN_SIZE1] = {0};
  float layer2[HIDDEN_SIZE2] = {0};


  // ===================================================
  // Layer 1 計算
  // ===================================================

  for (int i = 0; i < HIDDEN_SIZE1; i++) {

    layer1[i] = bias_1[i];

    for (int j = 0; j < INPUT_SIZE; j++) {

      layer1[i] +=
        weight_1[i * INPUT_SIZE + j]
        * input[j];
    }

    layer1[i] = relu(layer1[i]); // ReLU
  }


  // ===================================================
  // Layer 2 計算
  // ===================================================

  for (int i = 0; i < HIDDEN_SIZE2; i++) {

    layer2[i] = bias_2[i];

    for (int j = 0; j < HIDDEN_SIZE1; j++) {

      layer2[i] +=
        weight_2[i * HIDDEN_SIZE1 + j]
        * layer1[j];
    }

    layer2[i] = relu(layer2[i]); // ReLU
  }


  // ===================================================
  // Layer 3（出力層）計算
  // ===================================================

  for (int i = 0; i < OUTPUT_SIZE; i++) {

    output[i] = bias_3[i];

    for (int j = 0; j < HIDDEN_SIZE2; j++) {

      output[i] +=
        weight_3[i * HIDDEN_SIZE2 + j]
        * layer2[j];
    }

    // 出力層にはReLUを使用しない
  }
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
// D2：キャリブレーション
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
      "白を当ててD2をもう1回押してください"
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
      "ここからD3だけを使ってください"
    );

    Serial.println();

    return;
  }


  Serial.println(
    "キャリブレーション済みです。D3を使ってください。"
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
// D3：色読み取り + NN補正
// =====================================================

void handleRead() {

  // キャリブレーション前
  if (calibrationStep != 2) {

    Serial.println(
      "先にD2で黒→白のキャリブレーションをしてください"
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

        Serial.println(
          "1～6を入力してください"
        );
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

  initButton(
    calibButton,
    BUTTON_CALIB_PIN
  );


  initButton(
    readButton,
    BUTTON_READ_PIN
  );


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


  inputImageSize();


  Serial.println();

  Serial.println("============================");
  Serial.println("ニューラルネット版準備完了");
  Serial.println("============================");

  Serial.println();

  Serial.println("① 黒 → D2");
  Serial.println("② 白 → D2");
  Serial.println("③ その後はD3だけ");

  Serial.println();
}


// =====================================================
// loop
// =====================================================

void loop() {

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


  delay(1);
}