void outputPPM() {
  Serial.println("========== DISTRIBUTED CAPTURE START ==========");
  Serial.println("FIRMWARE=COLOR_RETRAIN_V7");
  Serial.print("CAPTURE=");Serial.println(captureNumber);
  Serial.println("INPUT_SOURCE=NEW_PHYSICAL_SENSOR_READINGS");
  Serial.println("FORMAT=DENSE_FP32_ONLY; NO_PRUNING_IN_THIS_STAGE");
  Serial.println("CHART_ROLE=LEARNING_3X3_REPEAT; NOT_INDEPENDENT_FINAL_CHART");
  Serial.println("REFERENCE_SOURCE=DISTRIBUTED_NOTEBOOK_CMYK_DISPLAY_RECOVERED");
  Serial.println("MODEL_STAGE=DISTRIBUTED_REFERENCE_RELIABILITY_LEARNING; NO_TEMPLATE_X");
  Serial.print("REFERENCE_FILE_SHA256=");Serial.println(trained_models::REFERENCE_SHA256);
  Serial.println("MODEL_COUNT=4; HIDDEN_SIZES=40,60,80,100");
  Serial.print("REFERENCE_NOTEBOOK_SHA256=");Serial.println(trained_models::REFERENCE_NOTEBOOK_SHA256);
  Serial.print("TRAINING_ROWS=");Serial.println(trained_models::TRAINING_ROWS);
  Serial.print("TRAINING_DATA_SHA256=");Serial.println(trained_models::DATA_SHA256);
  Serial.println("MSE_SCALE=RGB8_0_255; MEAN_OVER_9_PIXELS_AND_3_CHANNELS");
  Serial.println("MSE_GOAL_THIS_READ=300");
  Serial.println("mse_dataset,H,format,colors,mse");
  double mses[4]={0};
  for(int model=0;model<trained_models::MODEL_COUNT;model++) {
    double squaredError=0;
    for(int r=0;r<3;r++) for(int c=0;c<3;c++) for(int channel=0;channel<3;channel++) {
      int difference=static_cast<int>(trainedImages[model][r][c][channel])-trained_models::REFERENCE_RGB[r*3+c][channel];
      squaredError+=static_cast<double>(difference)*difference;
    }
    Serial.print("new_read_notebook_cmyk_reference,");Serial.print(trained_models::models[model].hidden);
    mses[model]=squaredError/27.0;
    Serial.print(",DENSE,9,");Serial.println(mses[model],6);
  }
  Serial.println("quality_goal,H,max_mse,met_for_this_read");
  for(int model=0;model<trained_models::MODEL_COUNT;model++) {
    Serial.print("new_reading_goal,");Serial.print(trained_models::models[model].hidden);
    Serial.print(",300,");Serial.println(mses[model]<=300.0?"YES":"NO");
  }
  for(int model=0;model<trained_models::MODEL_COUNT;model++) {
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
  Serial.println("=========== DISTRIBUTED CAPTURE END ===========");
  Serial.println("基準はreference_image1_cmyk_3x3.ppmです。次の新しい読取は n で黒・白の登録から開始します。");
}
