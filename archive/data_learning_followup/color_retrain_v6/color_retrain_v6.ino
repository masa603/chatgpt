#include <Arduino.h>
#include "trained_models_v6.h"
void outputNNInputs();

// =====================================================
// ニューラルネット設定
// =====================================================

#define INPUT_SIZE 3
#define OUTPUT_SIZE 3

// =====================================================
// スキャナ設定
// =====================================================

#define SAMPLE_COUNT 10
#define RGB_FLASH_DELAY 10
#define DEBOUNCE_MS 40

#define MAX_IMAGE_SIZE 3

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

int imageSize = 3;
uint8_t trainedImages[3][3][3][3];
float rawHistory[3][3][3];
unsigned long sampleCompletedMs[3][3];
unsigned int captureNumber=1;

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

void predict(float input[3],float output[3]) {
  float allOutputs[3][3];trained_models::inferAll(input,allOutputs);
  for(int model=0;model<3;model++) for(int c=0;c<3;c++) {
    float v=allOutputs[model][c];if(v<0) v=0;if(v>1) v=1;
    trainedImages[model][currentRow][currentColumn][c]=static_cast<int>(v*255.0+.5);
  }
  for(int c=0;c<3;c++) output[c]=allOutputs[1][c];
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
  Serial.println("========== LEARNING CAPTURE START ==========");
  Serial.println("FIRMWARE=COLOR_RETRAIN_V6");
  Serial.print("CAPTURE=");Serial.println(captureNumber);
  Serial.println("INPUT_SOURCE=NEW_PHYSICAL_SENSOR_READINGS");
  Serial.println("FORMAT=DENSE_FP32_ONLY; NO_PRUNING_IN_THIS_STAGE");
  Serial.println("CHART_ROLE=LEARNING_3X3_REPEAT; NOT_INDEPENDENT_FINAL_CHART");
  Serial.println("REFERENCE_SOURCE=USER_PROVIDED_result.ppm");
  Serial.println("MODEL_STAGE=DATA_ONLY_NEUTRAL_ADAPTATION_AFTER_V5; OLD_READINGS_NOW_TRAINING");
  Serial.print("REFERENCE_FILE_SHA256=");Serial.println(trained_models::REFERENCE_SHA256);
  Serial.print("TRAINING_ROWS=");Serial.println(trained_models::TRAINING_ROWS);
  Serial.print("TRAINING_DATA_SHA256=");Serial.println(trained_models::DATA_SHA256);
  Serial.println("MSE_SCALE=RGB8_0_255; MEAN_OVER_9_PIXELS_AND_3_CHANNELS");
  Serial.println("MSE_GOAL_THIS_READ=300");
  Serial.println("mse_dataset,H,format,colors,mse");
  double mses[3]={0};
  for(int model=0;model<3;model++) {
    double squaredError=0;
    for(int r=0;r<3;r++) for(int c=0;c<3;c++) for(int channel=0;channel<3;channel++) {
      int difference=static_cast<int>(trainedImages[model][r][c][channel])-trained_models::REFERENCE_RGB[r*3+c][channel];
      squaredError+=static_cast<double>(difference)*difference;
    }
    Serial.print("new_read_result_reference,");Serial.print(trained_models::models[model].hidden);
    mses[model]=squaredError/27.0;
    Serial.print(",DENSE,9,");Serial.println(mses[model],6);
  }
  Serial.println("quality_goal,H,max_mse,met_for_this_read");
  for(int model=0;model<3;model++) {
    Serial.print("new_reading_goal,");Serial.print(trained_models::models[model].hidden);
    Serial.print(",300,");Serial.println(mses[model]<=300.0?"YES":"NO");
  }
  for(int model=0;model<3;model++) {
    Serial.print("H=");Serial.println(trained_models::models[model].hidden);
    Serial.print("MODEL_SOURCE_SHA256=");Serial.println(trained_models::models[model].sha256);
    Serial.println("---------- PPM START ----------");
    Serial.println("P3");Serial.println("3 3");Serial.println("255");
    for(int r=0;r<3;r++) {
      for(int c=0;c<3;c++) for(int channel=0;channel<3;channel++) {
        Serial.print(trainedImages[model][r][c][channel]);Serial.print(" ");
      }
      Serial.println();
    }
    Serial.println("----------- PPM END -----------");
  }
  Serial.println("color_id,sample_completed_ms,raw_r,raw_g,raw_b");
  for(int r=0;r<3;r++) for(int c=0;c<3;c++) {
    Serial.print("r");Serial.print(r+1);Serial.print("c");Serial.print(c+1);Serial.print(",");Serial.print(sampleCompletedMs[r][c]);
    for(int channel=0;channel<3;channel++){Serial.print(",");Serial.print(rawHistory[r][c][channel],4);}
    Serial.println();
  }
  outputNNInputs();
  Serial.println("=========== LEARNING CAPTURE END ===========");
  Serial.println("基準は元のresult.ppmです。次の新しい読取は n で黒・白の登録から開始します。");
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
  sampleCompletedMs[currentRow][currentColumn]=millis();


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


  Serial.println("--- 60次元Dense補正後（全3モデルは9色後に表示）---");

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
    rawHistory[currentRow][currentColumn][channel]=rawRGB[channel];
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
    case 'n':
    case 'N':
      calibrationStep=0;currentRow=0;currentColumn=0;imageFinished=false;captureNumber++;
      Serial.println("新しい読取：黒 b → 白 w → 各色 r");
      break;
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




// =====================================================
// setup
// =====================================================

void setup() {

  Serial.begin(9600);

  delay(1000);
  Serial.println("FIRMWARE=COLOR_RETRAIN_V6");


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


  if(!trained_models::selfCheck()) { Serial.println("MODEL CHECK FAILED");while(true) delay(1000); }


  Serial.println();

  Serial.println("============================");
  Serial.println("灰色・白の追加学習版準備完了：3x3、画像サイズ入力不要");
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
