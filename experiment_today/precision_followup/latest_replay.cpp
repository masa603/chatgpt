// Inserted after definitions in precision_scan.ino. Uses archived sensor inputs, no sensing.
void printLatestReplay() {
  double squaredErrors[6] = {0};
  bool passed = true;
  float maxError = 0.0f;
  for (int sample = 0; sample < 9; sample++) {
    float output[6][3];
    fresh_models::inferAll(precision_replay::INPUTS[sample], output);
    for (int model = 0; model < 6; model++) {
      for (int c = 0; c < 3; c++) {
        float expected = precision_replay::EXPECTED[model * 9 + sample][c];
        float error = std::fabs(output[model][c] - expected);
        if (!std::isfinite(output[model][c]) || error > 1.0e-5f + 1.0e-5f * std::fabs(expected)) passed = false;
        if (error > maxError) maxError = error;
        int difference = outputTo255(output[model][c]) - precision_replay::TARGETS[sample][c];
        squaredErrors[model] += static_cast<double>(difference) * difference;
      }
    }
  }
  Serial.println("========== PRECISION REPLAY START ==========");
  Serial.println("FIRMWARE=PRECISION_SCAN_V2");
  Serial.println("PRUNING_STAGE=EXPLORATORY_PRECISION_FOLLOW_UP");
  Serial.println("INPUT_SOURCE=ARCHIVED_PHYSICAL_READINGS_FROM_OLD_FIRMWARE_RETRY_01");
  Serial.println("SCOPE=DEVICE_REPLAY_ON_SAVED_INPUTS; NOT_A_NEW_SENSOR_CAPTURE");
  Serial.println("CHART_ROLE=LEARNING_3X3_REPEAT; NOT_INDEPENDENT_FINAL_EVALUATION");
  Serial.println("TARGET_SOURCE=RECOVERED_NOTEBOOK_DISPLAY; ORIGINAL_PNG_NOT_VERIFIED");
  Serial.print("SOURCE_LOG_SHA256="); Serial.println(precision_replay::SOURCE_LOG_SHA256);
  for (int m = 0; m < 3; m++) {
    Serial.print("H="); Serial.println(fresh_models::models[m].hidden);
    Serial.print("THRESHOLD="); Serial.println(fresh_models::models[m].threshold);
    Serial.print("MODEL_SOURCE_SHA256="); Serial.println(fresh_models::models[m].sha256);
  }
  Serial.print("OUTPUT CHECK: "); Serial.println(passed ? "PASS" : "FAIL");
  Serial.print("MAX ABS ERROR="); Serial.println(maxError, 8);
  Serial.println("replay_dataset,H,format,colors,mse");
  for (int model = 0; model < 6; model++) {
    Serial.print("latest_physical_reading_replay,");
    Serial.print(fresh_models::models[model / 2].hidden); Serial.print(",");
    Serial.print(model % 2 == 0 ? "DENSE" : "CSR"); Serial.print(",9,");
    Serial.println(squaredErrors[model] / 27.0, 6);
  }
  Serial.println("=========== PRECISION REPLAY END ===========");
}
