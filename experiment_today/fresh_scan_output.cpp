// Inserted into fresh_scan_all.ino by build_fresh_scanner.py.
void outputPPM() {
  Serial.println("========== FRESH CAPTURE START ==========");
  Serial.print("CAPTURE="); Serial.println(captureNumber);
  Serial.println("INPUT_SOURCE=NEW_PHYSICAL_SENSOR_READINGS");
  Serial.println("CHART_ROLE=LEARNING_3X3_REPEAT; NOT_INDEPENDENT_FINAL_EVALUATION");
  Serial.println("COMPARISON=ALL_SIX_MODELS_USE_THE_SAME_FRESH_INPUTS");
  Serial.println("MODEL_GENERATION=CURRENT33_SEED42_EPOCH50000");
  Serial.println("SAMPLE_COUNT=10; RGB_FLASH_DELAY_MS=10; LED_DRIVE=ANALOG_WRITE_255_0");
  Serial.println("TARGET_SOURCE=RECOVERED_NOTEBOOK_DISPLAY; ORIGINAL_PNG_NOT_VERIFIED");
  Serial.println("fresh_mse_dataset,H,format,colors,mse");
  for (int model = 0; model < 6; model++) {
    double squaredError = 0.0;
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        for (int c = 0; c < 3; c++) {
          int difference = static_cast<int>(freshImages[model][row][col][c]) - fresh_models::REFERENCE_RGB[row * 3 + col][c];
          squaredError += static_cast<double>(difference) * difference;
        }
      }
    }
    Serial.print("learning_3x3_repeat,"); Serial.print(fresh_models::models[model / 2].hidden);
    Serial.print(","); Serial.print(model % 2 == 0 ? "DENSE" : "CSR");
    Serial.print(",9,"); Serial.println(squaredError / 27.0, 6);
  }
  for (int model = 0; model < 6; model++) {
    const fresh_models::Model &metadata = fresh_models::models[model / 2];
    Serial.print("H="); Serial.print(metadata.hidden); Serial.print(" MODE="); Serial.println(model % 2 == 0 ? "DENSE" : "CSR");
    Serial.print("MODEL_SOURCE_SHA256="); Serial.println(metadata.sha256);
    Serial.print("THRESHOLD="); Serial.println(model % 2 == 0 ? "0" : metadata.threshold);
    Serial.println("---------- PPM START ----------");
    Serial.println("P3"); Serial.println("3 3"); Serial.println("255");
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        for (int c = 0; c < 3; c++) { Serial.print(freshImages[model][row][col][c]); Serial.print(" "); }
      }
      Serial.println();
    }
    Serial.println("----------- PPM END -----------");
  }
  Serial.println("color_id,sample_completed_ms,raw_r,raw_g,raw_b");
  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      Serial.print("r"); Serial.print(row + 1); Serial.print("c"); Serial.print(col + 1);
      Serial.print(","); Serial.print(sampleCompletedMs[row][col]);
      for (int c = 0; c < 3; c++) { Serial.print(","); Serial.print(rawHistory[row][col][c], 4); }
      Serial.println();
    }
  }
  outputNNInputs();
  Serial.println("=========== FRESH CAPTURE END ===========");
  Serial.println("次の測定は n を送信し、黒・白の登録から繰り返してください。");
}
