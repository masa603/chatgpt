// Inserted by export_experiments.py. These functions run outside the timer.
bool validCSRLayer(const float *values, const uint8_t *columns,
                   const uint16_t *pointers, const float *bias,
                   int rows, int inputColumns, size_t nonzeros) {
  if (pointers[0] != 0 || pointers[rows] != nonzeros) return false;
  for (int row = 0; row < rows; row++) {
    if (!std::isfinite(bias[row]) || pointers[row] > pointers[row + 1]
        || pointers[row + 1] > nonzeros) return false;
    for (uint16_t p = pointers[row]; p < pointers[row + 1]; p++) {
      if (!std::isfinite(values[p]) || columns[p] >= inputColumns) return false;
      if (p > pointers[row] && columns[p - 1] >= columns[p]) return false;
    }
  }
  return true;
}

bool validCSRModel() {
  return validCSRLayer(weight_1_values, weight_1_col_idx, weight_1_row_ptr, bias_1,
                       CSR_HIDDEN_SIZE, 3, sizeof(weight_1_values) / sizeof(float))
      && validCSRLayer(weight_2_values, weight_2_col_idx, weight_2_row_ptr, bias_2,
                       CSR_HIDDEN_SIZE, CSR_HIDDEN_SIZE, sizeof(weight_2_values) / sizeof(float))
      && validCSRLayer(weight_3_values, weight_3_col_idx, weight_3_row_ptr, bias_3,
                       3, CSR_HIDDEN_SIZE, sizeof(weight_3_values) / sizeof(float));
}

int outputToRGB8(float value) {
  if (value < 0.0f) value = 0.0f;
  if (value > 1.0f) value = 1.0f;
  return static_cast<int>(value * 255.0f + 0.5f);
}

double savedInputMSE(void (*predict)(const float *, float *), const float (*inputs)[3],
                     const uint8_t (*targets)[3], uint16_t first, uint16_t count) {
  double squaredError = 0.0;
  for (uint16_t i = first; i < first + count; i++) {
    float output[3];
    predict(inputs[i], output);
    for (int c = 0; c < 3; c++) {
      if (!std::isfinite(output[c])) return NAN;
      int difference = outputToRGB8(output[c]) - static_cast<int>(targets[i][c]);
      squaredError += static_cast<double>(difference) * difference;
    }
  }
  return squaredError / (3.0 * count);
}

void printMSERow(const char *dataset, int readId, const float (*inputs)[3],
                 const uint8_t (*targets)[3], uint16_t first, uint16_t count) {
  double denseMSE = savedInputMSE(densePredict, inputs, targets, first, count);
  double csrMSE = savedInputMSE(csrPredict, inputs, targets, first, count);
  Serial.print(dataset); Serial.print(","); Serial.print(readId); Serial.print(",");
  Serial.print(count); Serial.print(","); Serial.print(denseMSE, 6);
  Serial.print(","); Serial.println(csrMSE, 6);
}

void printMSESummary() {
  Serial.println("MSE_SCALE=RGB8_0_255_HALF_UP");
  Serial.println("MSE_SOURCE=SAVED_INPUTS_REPLAYED_ON_DEVICE");
  Serial.println("TRAINING_TARGET_SOURCE=RECOVERED_NOTEBOOK_DISPLAY; ORIGINAL_PNG_NOT_VERIFIED");
  Serial.println("VALIDATION_ROLE=USED_TO_SELECT_THRESHOLD; NOT_INDEPENDENT_FINAL_TEST");
  Serial.println("INDEPENDENT_FINAL_EVALUATION=NOT_PERFORMED");
  Serial.println("mse_dataset,read_id,colors,dense_mse,csr_mse");
  printMSERow("learning_3x3", 0, TRAINING_INPUTS, TRAINING_TARGETS, 0, 9);
  printMSERow("learning_6x4", 0, TRAINING_INPUTS, TRAINING_TARGETS, 9, 24);
  printMSERow("learning_all33", 0, TRAINING_INPUTS, TRAINING_TARGETS, 0, TRAINING_INPUT_COUNT);
  for (uint16_t read = 0; read < VALIDATION_READ_COUNT; read++) {
    printMSERow("validation", read + 1, BENCHMARK_INPUTS, VALIDATION_TARGETS,
                read * VALIDATION_COLORS_PER_READ, VALIDATION_COLORS_PER_READ);
  }
  printMSERow("validation_mean", 0, BENCHMARK_INPUTS, VALIDATION_TARGETS, 0, BENCHMARK_INPUT_COUNT);
}
