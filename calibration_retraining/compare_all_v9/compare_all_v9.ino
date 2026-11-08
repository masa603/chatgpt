#include <Arduino.h>
#include <cmath>
#include "comparison_parameters_v9.h"
namespace trained_models {
void predict(int index, const float *input, float *output) {
  const Model &m=models[index];
  float layer1[100],layer2[100];
  for(int r=0;r<m.hidden;r++) {
    float value=m.b1[r];
    for(int c=0;c<3;c++) value+=m.w1[r*3+c]*input[c];
    layer1[r]=value>0.0f?value:0.0f;
  }
  for(int r=0;r<m.hidden;r++) {
    float value=m.b2[r];
    for(int c=0;c<m.hidden;c++) value+=m.w2[r*m.hidden+c]*layer1[c];
    layer2[r]=value>0.0f?value:0.0f;
  }
  for(int r=0;r<3;r++) {
    float value=m.b3[r];
    for(int c=0;c<m.hidden;c++) value+=m.w3[r*m.hidden+c]*layer2[c];
    output[r]=value;
  }
}

void inferAll(const float *input,float output[4][3]) {
  for(int i=0;i<MODEL_COUNT;i++) predict(i,input,output[i]);
}

bool selfCheck() {
  bool passed=true;
  Serial.println("========== RETRAIN CHECK START ==========");
  Serial.print("TRAINING_DATA_SHA256="); Serial.println(DATA_SHA256);
  Serial.print("TRAINING_ROWS="); Serial.println(TRAINING_ROWS);
  Serial.println("FORMAT=DENSE_FP32_ONLY");
  Serial.println("MODEL_COUNT=4; HIDDEN_SIZES=40,60,80,100");
  Serial.println("REFERENCE_SOURCE=DISTRIBUTED_NOTEBOOK_CMYK_DISPLAY_RECOVERED");
  Serial.print("REFERENCE_FILE_SHA256=");Serial.println(REFERENCE_SHA256);
  for(int i=0;i<MODEL_COUNT;i++) {
    float maxError=0;bool ok=true;
    for(int sample=0;sample<12;sample++) {
      float output[3];predict(i,CHECK_INPUTS[sample],output);
      for(int c=0;c<3;c++) {
        float expected=models[i].expected[sample][c];
        float error=std::fabs(output[c]-expected);
        if(!std::isfinite(output[c]) || error>1e-5f+1e-5f*std::fabs(expected)) ok=false;
        if(error>maxError) maxError=error;
      }
    }
    Serial.print("H="); Serial.println(models[i].hidden);
    Serial.print("MODEL_SOURCE_SHA256="); Serial.println(models[i].sha256);
    Serial.print("MODEL OUTPUT CHECK: ");Serial.println(ok?"PASS":"FAIL");
    Serial.print("MAX ABS ERROR=");Serial.println(maxError,8);
    passed=passed&&ok;
  }
  Serial.print("ALL MODELS CHECK: ");Serial.println(passed?"PASS":"FAIL");
  Serial.println("=========== RETRAIN CHECK END ===========");
  return passed;
}
}

namespace comparison_models {
void sparseLayer(const SparseLayer &layer,const float *input,float *output,bool relu) {
  for(int r=0;r<layer.rows;r++) {
    float value=layer.bias[r];
    for(uint16_t k=layer.row[r];k<layer.row[r+1];k++) value+=layer.values[k]*input[layer.col[k]];
    output[r]=(relu && value<0)?0:value;
  }
}

#if defined(__GNUC__)
__attribute__((noinline))
#endif
void predict(int model,const float *input,float *output) {
  if(model<4) { trained_models::predict(model,input,output);return; }
  float first[100],second[100];
  const SparseModel &m=sparse[model-4];
  sparseLayer(m.layers[0],input,first,true);
  sparseLayer(m.layers[1],first,second,true);
  sparseLayer(m.layers[2],second,output,false);
}

void inferAll(const float *input,float output[8][3]) {
  for(int model=0;model<MODEL_COUNT;model++) predict(model,input,output[model]);
}

bool validIndices(int index) {
  const SparseModel &model=sparse[index];
  for(int i=0;i<3;i++) {
    const SparseLayer &l=model.layers[i];
    if(l.row[0]!=0 || l.row[l.rows]!=l.nnz || l.stored<(l.nnz?l.nnz:1)) return false;
    for(int r=0;r<l.rows;r++) {
      if(l.row[r]>l.row[r+1] || l.row[r+1]>l.nnz) return false;
      for(uint16_t k=l.row[r];k<l.row[r+1];k++) {
        if(l.col[k]>=l.cols || !std::isfinite(l.values[k])) return false;
        if(k>l.row[r] && l.col[k]<=l.col[k-1]) return false;
      }
    }
  }
  return true;
}

bool selfCheckAll() {
  bool passed=trained_models::selfCheck();
  for(int index=0;index<4;index++) {
    bool ok=validIndices(index);float maxError=0;
    if(ok) for(int sample=0;sample<12;sample++) {
      float output[3];predict(index+4,trained_models::CHECK_INPUTS[sample],output);
      for(int c=0;c<3;c++) {
        float expected=sparse[index].expected[sample][c];
        float error=std::fabs(output[c]-expected);
        if(!std::isfinite(output[c]) || error>1e-5f+1e-5f*std::fabs(expected)) ok=false;
        if(error>maxError) maxError=error;
      }
    }
    Serial.print("CSR H=");Serial.print(trained_models::models[index].hidden);
    Serial.print(" INDEX_AND_OUTPUT_CHECK=");Serial.print(ok?"PASS":"FAIL");
    Serial.print(" MAX_ABS_ERROR=");Serial.println(maxError,8);
    passed=passed&&ok;
  }
  Serial.print("ALL EIGHT MODELS CHECK: ");Serial.println(passed?"PASS":"FAIL");
  return passed;
}
}

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
uint8_t trainedImages[8][3][3][3];
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
  float allOutputs[8][3];comparison_models::inferAll(input,allOutputs);
  for(int model=0;model<comparison_models::MODEL_COUNT;model++) for(int c=0;c<3;c++) {
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
  Serial.println("========== PAIRED CAPTURE START ==========");
  Serial.println("FIRMWARE=COLOR_COMPARE_V9");
  Serial.print("CAPTURE=");Serial.println(captureNumber);
  Serial.println("INPUT_SOURCE=NEW_PHYSICAL_SENSOR_READINGS");
  Serial.println("COMPARISON=ALL_EIGHT_CONDITIONS_USE_THE_SAME_NEW_INPUTS");
  Serial.println("CHART_ROLE=LEARNING_3X3_REPEAT; NOT_INDEPENDENT_FINAL_CHART");
  Serial.println("REFERENCE_SOURCE=DISTRIBUTED_NOTEBOOK_CMYK_DISPLAY_RECOVERED");
  Serial.print("REFERENCE_FILE_SHA256=");Serial.println(trained_models::REFERENCE_SHA256);
  Serial.print("TRAINING_DATA_SHA256=");Serial.println(trained_models::DATA_SHA256);
  Serial.println("TRAINING_ROWS=135; MODEL_GENERATION=FROZEN_V9");
  Serial.println("CONDITIONS=H40_H60_H80_H100; EACH_DENSE_AND_PRUNE_CSR");
  Serial.println("CSR_SELECTION=ARCHIVED_SEPARATE_VALIDATION_2_READS; PROVISIONAL_INPUT_COLLISIONS");
  Serial.println("INDEPENDENT_FINAL_EVALUATION=NOT_PERFORMED");
  Serial.println("mse_dataset,H,format,colors,mse");
  double mses[8]={0};
  for(int model=0;model<comparison_models::MODEL_COUNT;model++) {
    double squaredError=0;
    for(int r=0;r<3;r++) for(int c=0;c<3;c++) for(int channel=0;channel<3;channel++) {
      int d=static_cast<int>(trainedImages[model][r][c][channel])-trained_models::REFERENCE_RGB[r*3+c][channel];
      squaredError+=static_cast<double>(d)*d;
    }
    mses[model]=squaredError/27.0;
    Serial.print("new_learning_chart_read,");Serial.print(trained_models::models[model%4].hidden);
    Serial.print(model<4?",DENSE,9,":",CSR,9,");Serial.println(mses[model],6);
  }
  Serial.println("quality_goal,H,format,max_mse,met_for_this_read");
  for(int model=0;model<comparison_models::MODEL_COUNT;model++) {
    Serial.print("user_accuracy_goal,");Serial.print(trained_models::models[model%4].hidden);
    Serial.print(model<4?",DENSE,300,":",CSR,300,");Serial.println(mses[model]<=300?"YES":"NO");
  }
  Serial.println("H,dense_mse,csr_mse,csr_minus_dense,change_percent_if_dense_nonzero");
  for(int i=0;i<4;i++) {
    Serial.print(trained_models::models[i].hidden);Serial.print(",");Serial.print(mses[i],6);
    Serial.print(",");Serial.print(mses[i+4],6);Serial.print(",");Serial.print(mses[i+4]-mses[i],6);
    Serial.print(",");if(mses[i]>0) Serial.println((mses[i+4]/mses[i]-1)*100,6);else Serial.println("NA");
  }
  for(int model=0;model<comparison_models::MODEL_COUNT;model++) {
    int base=model%4;
    Serial.print("H=");Serial.println(trained_models::models[base].hidden);
    Serial.print("FORMAT=");Serial.println(model<4?"DENSE":"CSR");
    Serial.print("MODEL_SOURCE_SHA256=");Serial.println(trained_models::models[base].sha256);
    Serial.print("THRESHOLD=");Serial.println(model<4?0:comparison_models::sparse[base].threshold,17);
    Serial.println("---------- PPM START ----------");
    Serial.println("P3");Serial.println("3 3");Serial.println("255");
    for(int r=0;r<3;r++) {
      for(int c=0;c<3;c++) for(int channel=0;channel<3;channel++) {Serial.print(trainedImages[model][r][c][channel]);Serial.print(" ");}
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
  Serial.println("=========== PAIRED CAPTURE END ===========");
  Serial.println("速度・容量比較は t、次の9色読取は n を送信してください。");
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


  Serial.println("--- 60次元Dense補正後（全8条件は9色後に表示）---");

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

#if defined(__GNUC__)
__attribute__((noinline))
#endif
uint32_t timeModel(int model,int group,double &checksum) {
  float output[3];checksum=0;
  uint32_t begin=static_cast<uint32_t>(micros());
  for(int call=0;call<100;call++) {
    int input=(group*100+call)%comparison_models::BENCHMARK_INPUT_COUNT;
    comparison_models::predict(model,comparison_models::BENCHMARK_INPUTS[input],output);
    checksum+=output[0]+output[1]+output[2];
  }
  return static_cast<uint32_t>(micros())-begin;
}

void runBenchmarks() {
  if(!comparison_models::selfCheckAll()) {Serial.println("BENCHMARK REFUSED: MODEL CHECK FAILED");return;}
  Serial.println("========== FOUR PAIR BENCHMARK START ==========");
  Serial.println("FIRMWARE=COLOR_COMPARE_V9; MODEL_GENERATION=FROZEN_V9");
  Serial.println("INPUT_SOURCE=ARCHIVED_VALIDATION; NO_NEW_SENSOR_CAPTURE");
  Serial.print("INPUT_SHA256=");Serial.println(comparison_models::BENCHMARK_INPUT_SHA256);
  Serial.println("INPUT_COUNT=128; WARMUP_CALLS=10; CALLS_PER_GROUP=100; GROUPS=5");
  Serial.println("TIMING_SCOPE=PREDICTION_AND_COMMON_CHECKSUM_ACCUMULATION; SENSOR_SERIAL_MSE_EXCLUDED");
  Serial.println("MODEL_BYTES_SCOPE=NUMERIC_ARRAYS_INCLUDING_BIAS_AND_CSR_INDICES; NOT_WHOLE_FIRMWARE_FLASH_OR_RAM");
  Serial.println("DECLARED_PREDICTION_WORK_ARRAY_BYTES=800; STACK_TOTAL_NOT_MEASURED");
  Serial.println("CSR_SELECTION=PROVISIONAL; INDEPENDENT_FINAL_EVALUATION=NOT_PERFORMED");
  Serial.print("COMPILER=");Serial.println(__VERSION__);
#ifdef F_CPU
  Serial.print("F_CPU_HZ=");Serial.println(F_CPU);
#endif
  for(int i=0;i<4;i++) {
    Serial.print("H=");Serial.println(trained_models::models[i].hidden);
    Serial.print("MODEL_SOURCE_SHA256=");Serial.println(trained_models::models[i].sha256);
    Serial.print("THRESHOLD=");Serial.println(comparison_models::sparse[i].threshold,17);
    Serial.print("DENSE_MODEL_BYTES=");Serial.println(comparison_models::DENSE_BYTES[i]);
    Serial.print("CSR_MODEL_BYTES=");Serial.println(comparison_models::sparse[i].bytes);
    float warm[3];double warmChecksum=0;
    for(int call=0;call<10;call++) for(int format=0;format<2;format++) {
      comparison_models::predict(i+format*4,comparison_models::BENCHMARK_INPUTS[call],warm);
      warmChecksum+=warm[0]+warm[1]+warm[2];
    }
    double denseTimes[5],csrTimes[5];bool valid=true;
    Serial.println("group,first_model,dense_total_us,csr_total_us,dense_us_per_inference,csr_us_per_inference,dense_checksum,csr_checksum");
    for(int group=0;group<5;group++) {
      uint32_t denseTotal,csrTotal;double denseChecksum,csrChecksum;
      bool denseFirst=(group%2)==0;
      if(denseFirst) {denseTotal=timeModel(i,group,denseChecksum);csrTotal=timeModel(i+4,group,csrChecksum);}
      else {csrTotal=timeModel(i+4,group,csrChecksum);denseTotal=timeModel(i,group,denseChecksum);}
      denseTimes[group]=denseTotal/100.0;csrTimes[group]=csrTotal/100.0;
      valid=valid && denseTotal>0 && csrTotal>0 && std::isfinite(denseChecksum) && std::isfinite(csrChecksum);
      Serial.print(group+1);Serial.print(denseFirst?",Dense,":",CSR,");Serial.print(denseTotal);
      Serial.print(",");Serial.print(csrTotal);Serial.print(",");Serial.print(denseTimes[group],4);
      Serial.print(",");Serial.print(csrTimes[group],4);Serial.print(",");Serial.print(denseChecksum,8);
      Serial.print(",");Serial.println(csrChecksum,8);
    }
    double denseMean=0,csrMean=0;
    for(int j=0;j<5;j++){denseMean+=denseTimes[j]/5;csrMean+=csrTimes[j]/5;}
    double denseVariance=0,csrVariance=0;
    for(int j=0;j<5;j++){denseVariance+=(denseTimes[j]-denseMean)*(denseTimes[j]-denseMean)/4;csrVariance+=(csrTimes[j]-csrMean)*(csrTimes[j]-csrMean)/4;}
    Serial.print("DENSE mean_us=");Serial.print(denseMean,4);Serial.print(" sample_stddev_us=");Serial.println(std::sqrt(denseVariance),4);
    Serial.print("CSR mean_us=");Serial.print(csrMean,4);Serial.print(" sample_stddev_us=");Serial.println(std::sqrt(csrVariance),4);
    Serial.print("CSR time_change_percent=");if(denseMean>0)Serial.println((csrMean/denseMean-1)*100,4);else Serial.println("NA");
    Serial.print("WARMUP_CHECKSUM=");Serial.println(warmChecksum,8);
    Serial.print("TIMING VALID: ");Serial.println(valid?"YES":"NO");
  }
  Serial.println("=========== FOUR PAIR BENCHMARK END ===========");
  Serial.println("Arduino IDEのFlash・グローバル変数使用量は、この8条件入りプログラム全体の値として保存してください。");
}

void handleSerialCommand() {
  if (Serial.available() <= 0) {
    return;
  }

  int command = Serial.read();

  switch (command) {
    case 't':
    case 'T':
      runBenchmarks();return;
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
  Serial.println("FIRMWARE=COLOR_COMPARE_V9");


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


  if(!comparison_models::selfCheckAll()) { Serial.println("MODEL CHECK FAILED");while(true) delay(1000); }


  Serial.println();

  Serial.println("============================");
  Serial.println("V9のDense/CSR・全8条件準備完了：3x3、画像サイズ入力不要");
  Serial.println("============================");

  Serial.println();

  Serial.println("① 黒を当てて b を送信");
  Serial.println("② 白を当てて w を送信");
  Serial.println("③ 各色を当てて r を1回ずつ送信。速度・容量比較は t");
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
