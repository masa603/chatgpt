// Change these two settings, then click Verify for each row in the README.
#ifndef PROBE_H
#define PROBE_H 40
#endif
#ifndef PROBE_CSR
#define PROBE_CSR 0  // 0 = Dense FP32, 1 = pruned CSR FP32
#endif

#include <Arduino.h>
#include "model_data.h"
#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif
void modelPredict(const float *input, float *output);

// Volatile inputs prevent the model calculation being folded into constants.
// Volatile outputs keep inference results observable even without serial traffic.
volatile float probeInput[3] = {0.2f, 0.4f, 0.6f};
volatile float probeOutput[3] = {0.0f, 0.0f, 0.0f};
#if PROBE_CSR == 0
NOINLINE void modelPredict(const float *input, float *output) {
  float layer1[CSR_HIDDEN_SIZE] = {0};
  float layer2[CSR_HIDDEN_SIZE] = {0};
  for (int row = 0; row < CSR_HIDDEN_SIZE; row++) {
    float sum = bias_1[row];
    for (int col = 0; col < 3; col++) sum += dense_weight_1[row * 3 + col] * input[col];
    layer1[row] = sum > 0.0f ? sum : 0.0f;
  }
  for (int row = 0; row < CSR_HIDDEN_SIZE; row++) {
    float sum = bias_2[row];
    for (int col = 0; col < CSR_HIDDEN_SIZE; col++) sum += dense_weight_2[row * CSR_HIDDEN_SIZE + col] * layer1[col];
    layer2[row] = sum > 0.0f ? sum : 0.0f;
  }
  for (int row = 0; row < 3; row++) {
    float sum = bias_3[row];
    for (int col = 0; col < CSR_HIDDEN_SIZE; col++) sum += dense_weight_3[row * CSR_HIDDEN_SIZE + col] * layer2[col];
    output[row] = sum;
  }
}
#else
void sparseLayer(const float *input, const float *values, const uint8_t *columns,
                 const uint16_t *pointers, const float *bias, int rows,
                 float *output, bool useRelu) {
  for (int row = 0; row < rows; row++) {
    float sum = bias[row];
    for (uint16_t p = pointers[row]; p < pointers[row + 1]; p++) sum += values[p] * input[columns[p]];
    output[row] = useRelu ? (sum > 0.0f ? sum : 0.0f) : sum;
  }
}
NOINLINE void modelPredict(const float *input, float *output) {
  float layer1[CSR_HIDDEN_SIZE] = {0};
  float layer2[CSR_HIDDEN_SIZE] = {0};
  sparseLayer(input, weight_1_values, weight_1_col_idx, weight_1_row_ptr, bias_1, CSR_HIDDEN_SIZE, layer1, true);
  sparseLayer(layer1, weight_2_values, weight_2_col_idx, weight_2_row_ptr, bias_2, CSR_HIDDEN_SIZE, layer2, true);
  sparseLayer(layer2, weight_3_values, weight_3_col_idx, weight_3_row_ptr, bias_3, 3, output, false);
}
#endif

void setup() {
  Serial.begin(9600);
  delay(1000);
  Serial.println("SINGLE_MODEL_MEMORY_PROBE");
  Serial.print("H="); Serial.println(PROBE_H);
  Serial.print("FORMAT_ID="); Serial.println(PROBE_CSR);
  Serial.print("MODEL_SOURCE_SHA256="); Serial.println(CSR_SOURCE_MODEL_SHA256);
  Serial.print("MODEL_ARRAY_BYTES="); Serial.println(PROBE_MODEL_ARRAY_BYTES);
}

void loop() {
  float input[3] = {probeInput[0], probeInput[1], probeInput[2]};
  float output[3];
  modelPredict(input, output);
  for (int c = 0; c < 3; c++) probeOutput[c] = output[c];
  delay(100);
}
