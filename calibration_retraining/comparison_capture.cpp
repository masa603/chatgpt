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
