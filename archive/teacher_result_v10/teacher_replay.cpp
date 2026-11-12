bool printTeacherReferenceResults() {
  uint8_t pixels[8][9][3];double mse[8]={0};bool matching=true;
  for(int point=0;point<9;point++) {
    float output[8][3];comparison_models::inferAll(comparison_models::BENCHMARK_INPUTS[point],output);
    for(int model=0;model<8;model++)for(int c=0;c<3;c++) {
      float v=output[model][c];if(!std::isfinite(v)){Serial.println("NONFINITE OUTPUT");return false;}
      if(v<0)v=0;if(v>1)v=1;
      pixels[model][point][c]=static_cast<int>(v*255.0+.5);
      matching=matching && pixels[model][point][c]==evaluation_data::EXPECTED_RGB[model*9+point][c];
      int e=static_cast<int>(pixels[model][point][c])-trained_models::REFERENCE_RGB[point][c];
      mse[model]+=static_cast<double>(e)*e/27.0;
    }
  }
  if(!matching){Serial.println("REPLAY REFUSED: RGB DOES NOT MATCH FROZEN NUMERICAL REFERENCE");return false;}
  Serial.println("========== TEACHER RESULT COMPARISON START ==========");
  Serial.println("FIRMWARE=TEACHER_RESULT_COMPARE_V10");
  Serial.println("REFERENCE_SOURCE=TEACHER_REQUIRED_ORIGINAL_RESULT_PPM");
  Serial.print("REFERENCE_FILE_SHA256=");Serial.println(trained_models::REFERENCE_SHA256);
  Serial.print("TRAINING_DATA_SHA256=");Serial.println(trained_models::DATA_SHA256);
  Serial.println("TRAINING_ROWS=135; HELDOUT_INPUT_COUNT=9; HELDOUT_INPUTS_NOT_FITTED");
  Serial.println("INPUT_SOURCE=ARCHIVED_REAL_SENSOR_READINGS; NO_NEW_SENSOR_CAPTURE");
  Serial.println("EVALUATION_ROLE=RETROSPECTIVE_SAME_CHART_READING; NOT_INDEPENDENT_FINAL_CHART");
  Serial.print("SOURCE_PHYSICAL_LOG_SHA256=");Serial.println(evaluation_data::SOURCE_PHYSICAL_LOG_SHA256);
  Serial.println("ALL EIGHT RGB NUMERICAL CHECK: PASS");
  Serial.println("mse_dataset,H,format,colors,mse");
  for(int model=0;model<8;model++) {
    Serial.print("teacher_result_heldout9,");Serial.print(trained_models::models[model%4].hidden);
    Serial.print(model<4?",DENSE,9,":",CSR,9,");Serial.println(mse[model],6);
  }
  Serial.println("quality_goal,H,format,max_mse,met_on_saved_input");
  for(int model=0;model<8;model++) {
    Serial.print("user_accuracy_goal,");Serial.print(trained_models::models[model%4].hidden);
    Serial.print(model<4?",DENSE,300,":",CSR,300,");Serial.println(mse[model]<=300?"YES":"NO");
  }
  Serial.println("H,dense_mse,csr_mse,csr_minus_dense,change_percent_if_dense_nonzero");
  for(int i=0;i<4;i++) {
    Serial.print(trained_models::models[i].hidden);Serial.print(",");Serial.print(mse[i],6);
    Serial.print(",");Serial.print(mse[i+4],6);Serial.print(",");Serial.print(mse[i+4]-mse[i],6);
    Serial.print(",");if(mse[i]>0)Serial.println((mse[i+4]/mse[i]-1)*100,6);else Serial.println("NA");
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
  Serial.println("=========== TEACHER RESULT COMPARISON END ===========");
  return true;
}
