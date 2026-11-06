// Appended to the generated models_all.h; all calculations use frozen FP32 weights.
namespace fresh_models {
struct Layer {
  const float *dense;
  const float *bias;
  const float *values;
  const uint8_t *columns;
  const uint16_t *pointers;
  int rows;
  int inputColumns;
  size_t nonzeros;
};
struct Model {
  int hidden;
  const char *threshold;
  const char *sha256;
  size_t denseBytes;
  size_t csrBytes;
  const float (*denseExpected)[3];
  const float (*csrExpected)[3];
  Layer layers[3];
};
// The generator inserts models[], CHECK_INPUTS and reference RGB before these functions.
// INSERT_MODEL_DEFINITIONS

void denseLayer(const Layer &layer, const float *input, float *output, bool relu) {
  for (int row = 0; row < layer.rows; row++) {
    float sum = layer.bias[row];
    for (int col = 0; col < layer.inputColumns; col++) sum += layer.dense[row * layer.inputColumns + col] * input[col];
    output[row] = relu ? (sum > 0.0f ? sum : 0.0f) : sum;
  }
}

void sparseLayer(const Layer &layer, const float *input, float *output, bool relu) {
  for (int row = 0; row < layer.rows; row++) {
    float sum = layer.bias[row];
    for (uint16_t p = layer.pointers[row]; p < layer.pointers[row + 1]; p++) sum += layer.values[p] * input[layer.columns[p]];
    output[row] = relu ? (sum > 0.0f ? sum : 0.0f) : sum;
  }
}

void densePredict(int modelIndex, const float *input, float *output) {
  float l1[100], l2[100];
  const Model &model = models[modelIndex];
  denseLayer(model.layers[0], input, l1, true);
  denseLayer(model.layers[1], l1, l2, true);
  denseLayer(model.layers[2], l2, output, false);
}

void csrPredict(int modelIndex, const float *input, float *output) {
  float l1[100], l2[100];
  const Model &model = models[modelIndex];
  sparseLayer(model.layers[0], input, l1, true);
  sparseLayer(model.layers[1], l1, l2, true);
  sparseLayer(model.layers[2], l2, output, false);
}

void inferAll(const float *input, float output[6][3]) {
  for (int model = 0; model < 3; model++) {
    densePredict(model, input, output[2 * model]);
    csrPredict(model, input, output[2 * model + 1]);
  }
}

bool validModel(const Model &model) {
  for (int i = 0; i < 3; i++) {
    const Layer &l = model.layers[i];
    if (l.pointers[0] != 0 || l.pointers[l.rows] != l.nonzeros) return false;
    for (int r = 0; r < l.rows; r++) {
      if (!std::isfinite(l.bias[r]) || l.pointers[r] > l.pointers[r + 1] || l.pointers[r + 1] > l.nonzeros) return false;
      for (uint16_t p = l.pointers[r]; p < l.pointers[r + 1]; p++) {
        if (!std::isfinite(l.values[p]) || l.columns[p] >= l.inputColumns) return false;
        if (p > l.pointers[r] && l.columns[p - 1] >= l.columns[p]) return false;
      }
    }
  }
  return true;
}

bool selfCheck() {
  bool passed = true;
  Serial.println("========== FRESH SCAN CHECK START ==========");
  for (int model = 0; model < 3; model++) {
    const Model &m = models[model];
    Serial.print("H="); Serial.println(m.hidden);
    Serial.print("MODEL_SOURCE_SHA256="); Serial.println(m.sha256);
    Serial.print("THRESHOLD="); Serial.println(m.threshold);
    Serial.print("MODEL BYTES: Dense="); Serial.print(m.denseBytes); Serial.print(" CSR="); Serial.println(m.csrBytes);
    bool valid = validModel(m);
    bool denseOK = valid, csrOK = valid;
    float maxError[2] = {0.0f, 0.0f};
    if (valid) {
      for (int sample = 0; sample < 12; sample++) {
        float output[2][3];
        densePredict(model, CHECK_INPUTS[sample], output[0]);
        csrPredict(model, CHECK_INPUTS[sample], output[1]);
        for (int mode = 0; mode < 2; mode++) {
          const float *expected = mode == 0 ? m.denseExpected[sample] : m.csrExpected[sample];
          for (int c = 0; c < 3; c++) {
            float error = std::fabs(output[mode][c] - expected[c]);
            bool correct = std::isfinite(output[mode][c]) && error <= 1.0e-5f + 1.0e-5f * std::fabs(expected[c]);
            if (!correct) { if (mode == 0) denseOK = false; else csrOK = false; }
            if (error > maxError[mode]) maxError[mode] = error;
          }
        }
      }
    }
    Serial.print("CSR INDEX CHECK: "); Serial.println(valid ? "PASS" : "FAIL");
    Serial.print("FIXED INPUT CHECK: Dense="); Serial.print(denseOK ? "PASS" : "FAIL");
    Serial.print(" CSR="); Serial.println(csrOK ? "PASS" : "FAIL");
    Serial.print("MAX ABS ERROR: Dense="); Serial.print(maxError[0], 8); Serial.print(" CSR="); Serial.println(maxError[1], 8);
    passed = passed && denseOK && csrOK;
  }
  Serial.print("ALL MODELS CHECK: "); Serial.println(passed ? "PASS" : "FAIL");
  Serial.println("=========== FRESH SCAN CHECK END ===========");
  return passed;
}
}  // namespace fresh_models
