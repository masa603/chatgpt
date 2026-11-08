bool replaySavedCapture() {
  uint8_t pixels[8][9][3];
  double mses[8]={0};
  double pureReferenceMses[8]={0};
  bool denseMatches=true;
  for(int point=0;point<9;point++) {
    float output[8][3];
    comparison_models::inferAll(saved_capture::INPUTS[point],output);
    for(int model=0;model<8;model++) for(int c=0;c<3;c++) {
      float value=output[model][c];
      if(!std::isfinite(value)) {Serial.println("SAVED REPLAY REFUSED: NONFINITE OUTPUT");return false;}
      if(value<0)value=0;if(value>1)value=1;
      pixels[model][point][c]=static_cast<int>(value*255.0+.5);
      int error=static_cast<int>(pixels[model][point][c])-trained_models::REFERENCE_RGB[point][c];
      mses[model]+=static_cast<double>(error)*error/27.0;
      int pureError=static_cast<int>(pixels[model][point][c])-saved_capture::SITE_RESULT_PURE_RGB[point][c];
      pureReferenceMses[model]+=static_cast<double>(pureError)*pureError/27.0;
      if(model<4 && pixels[model][point][c]!=saved_capture::ORIGINAL_DENSE_RGB[model*9+point][c]) denseMatches=false;
    }
  }
  if(!denseMatches) {Serial.println("SAVED REPLAY REFUSED: DENSE RGB DOES NOT MATCH RECEIVED CAPTURE");return false;}
  Serial.println("========== SAVED CAPTURE COMPARISON START ==========");
  Serial.println("FIRMWARE=COLOR_SAVED_COMPARE_V9");
  Serial.println("INPUT_SOURCE=ARCHIVED_NEW_PHYSICAL_COLOR_RETRAIN_V9_CAPTURE1");
  Serial.println("SCOPE=DEVICE_REPLAY; NO_NEW_SENSOR_CAPTURE; PHYSICAL_CAPTURE_COUNT=1");
  Serial.println("CHART_ROLE=LEARNING_3X3_REPEAT; NOT_INDEPENDENT_FINAL_EVALUATION");
  Serial.println("THRESHOLDS_FROZEN_BEFORE_SOURCE_CAPTURE; NO_MODEL_OR_THRESHOLD_UPDATE");
  Serial.print("SOURCE_LOG_SHA256=");Serial.println(saved_capture::SOURCE_LOG_SHA256);
  Serial.print("SAVED_INPUT_CSV_SHA256=");Serial.println(saved_capture::INPUT_CSV_SHA256);
  Serial.print("REFERENCE_FILE_SHA256=");Serial.println(trained_models::REFERENCE_SHA256);
  Serial.println("PRIMARY_REFERENCE=DISTRIBUTED_NOTEBOOK_CMYK; USER_SELECTED_TRAINING_TARGET");
  Serial.print("SITE_RESULT_PURE_RGB_REFERENCE_SHA256=");Serial.println(saved_capture::SITE_RESULT_REFERENCE_SHA256);
  Serial.println("ORIGINAL_DENSE_RGB_CHECK: EXACT_MATCH");
  Serial.println("mse_dataset,H,format,colors,mse");
  for(int model=0;model<8;model++) {
    Serial.print("saved_v9_capture1,");Serial.print(trained_models::models[model%4].hidden);
    Serial.print(model<4?",DENSE,9,":",CSR,9,");Serial.println(mses[model],6);
  }
  Serial.println("H,format,notebook_cmyk_mse,site_result_pure_rgb_mse");
  for(int model=0;model<8;model++) {
    Serial.print(trained_models::models[model%4].hidden);Serial.print(model<4?",DENSE,":",CSR,");
    Serial.print(mses[model],6);Serial.print(",");Serial.println(pureReferenceMses[model],6);
  }
  Serial.println("USER_GOAL_REFERENCE=NOTEBOOK_CMYK; PURE_RGB_IS_A_DIFFERENT_COMPARISON");
  Serial.println("quality_goal,H,format,max_mse,met_for_saved_input");
  for(int model=0;model<8;model++) {
    Serial.print("user_accuracy_goal,");Serial.print(trained_models::models[model%4].hidden);
    Serial.print(model<4?",DENSE,300,":",CSR,300,");Serial.println(mses[model]<=300?"YES":"NO");
  }
  Serial.println("H,dense_mse,csr_mse,csr_minus_dense,change_percent_if_dense_nonzero,increase_within5percent_on_saved_input");
  for(int i=0;i<4;i++) {
    Serial.print(trained_models::models[i].hidden);Serial.print(",");Serial.print(mses[i],6);
    Serial.print(",");Serial.print(mses[i+4],6);Serial.print(",");Serial.print(mses[i+4]-mses[i],6);
    Serial.print(",");if(mses[i]>0)Serial.print((mses[i+4]/mses[i]-1)*100,6);else Serial.print("NA");
    Serial.print(",");Serial.println(mses[i+4]<=mses[i]*1.05?"YES":"NO");
  }
  for(int model=0;model<8;model++) {
    int base=model%4;
    Serial.print("H=");Serial.println(trained_models::models[base].hidden);
    Serial.print("FORMAT=");Serial.println(model<4?"DENSE":"CSR");
    Serial.print("MODEL_SOURCE_SHA256=");Serial.println(trained_models::models[base].sha256);
    Serial.print("THRESHOLD=");Serial.println(model<4?0:comparison_models::sparse[base].threshold,17);
    Serial.println("---------- PPM START ----------");Serial.println("P3");Serial.println("3 3");Serial.println("255");
    for(int r=0;r<3;r++) {
      for(int col=0;col<3;col++)for(int c=0;c<3;c++){Serial.print(pixels[model][r*3+col][c]);Serial.print(" ");}
      Serial.println();
    }
    Serial.println("----------- PPM END -----------");
  }
  Serial.println("=========== SAVED CAPTURE COMPARISON END ===========");
  return true;
}
