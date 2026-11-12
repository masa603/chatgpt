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
