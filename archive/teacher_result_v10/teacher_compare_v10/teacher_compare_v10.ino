#include <Arduino.h>
#include <cmath>
#include "teacher_parameters_v10.h"

namespace trained_models {
void predict(int index, const float *input, float *output) {
  const Model &m=models[index];
  float layer1[100],layer2[100];
  for(int r=0;r<m.hidden;r++) {
    float value=m.b1[r];
    for(int c=0;c<3;c++) value+=m.w1[r*3+c]*input[c];
    layer1[r]=value>0.0f?value:0.0f;
  }
  for(int r=0;r<m.hidden;r++) {
    float value=m.b2[r];
    for(int c=0;c<m.hidden;c++) value+=m.w2[r*m.hidden+c]*layer1[c];
    layer2[r]=value>0.0f?value:0.0f;
  }
  for(int r=0;r<3;r++) {
    float value=m.b3[r];
    for(int c=0;c<m.hidden;c++) value+=m.w3[r*m.hidden+c]*layer2[c];
    output[r]=value;
  }
}

void inferAll(const float *input,float output[4][3]) {
  for(int i=0;i<MODEL_COUNT;i++) predict(i,input,output[i]);
}

bool selfCheck() {
  bool passed=true;
  Serial.println("========== RETRAIN CHECK START ==========");
  Serial.print("TRAINING_DATA_SHA256="); Serial.println(DATA_SHA256);
  Serial.print("TRAINING_ROWS="); Serial.println(TRAINING_ROWS);
  Serial.println("FORMAT=DENSE_FP32_ONLY");
  Serial.println("MODEL_COUNT=4; HIDDEN_SIZES=40,60,80,100");
  Serial.println("REFERENCE_SOURCE=TEACHER_REQUIRED_ORIGINAL_RESULT_PPM");
  Serial.print("REFERENCE_FILE_SHA256=");Serial.println(REFERENCE_SHA256);
  for(int i=0;i<MODEL_COUNT;i++) {
    float maxError=0;bool ok=true;
    for(int sample=0;sample<12;sample++) {
      float output[3];predict(i,CHECK_INPUTS[sample],output);
      for(int c=0;c<3;c++) {
        float expected=models[i].expected[sample][c];
        float error=std::fabs(output[c]-expected);
        if(!std::isfinite(output[c]) || error>1e-5f+1e-5f*std::fabs(expected)) ok=false;
        if(error>maxError) maxError=error;
      }
    }
    Serial.print("H="); Serial.println(models[i].hidden);
    Serial.print("MODEL_SOURCE_SHA256="); Serial.println(models[i].sha256);
    Serial.print("MODEL OUTPUT CHECK: ");Serial.println(ok?"PASS":"FAIL");
    Serial.print("MAX ABS ERROR=");Serial.println(maxError,8);
    passed=passed&&ok;
  }
  Serial.print("ALL MODELS CHECK: ");Serial.println(passed?"PASS":"FAIL");
  Serial.println("=========== RETRAIN CHECK END ===========");
  return passed;
}
}

namespace comparison_models {
void sparseLayer(const comparison_models::SparseLayer &layer,const float *input,float *output,bool relu) {
  for(int r=0;r<layer.rows;r++) {
    float value=layer.bias[r];
    for(uint16_t k=layer.row[r];k<layer.row[r+1];k++) value+=layer.values[k]*input[layer.col[k]];
    output[r]=(relu && value<0)?0:value;
  }
}

#if defined(__GNUC__)
__attribute__((noinline))
#endif
void predict(int model,const float *input,float *output) {
  if(model<4) { trained_models::predict(model,input,output);return; }
  float first[100],second[100];
  const SparseModel &m=sparse[model-4];
  sparseLayer(m.layers[0],input,first,true);
  sparseLayer(m.layers[1],first,second,true);
  sparseLayer(m.layers[2],second,output,false);
}

void inferAll(const float *input,float output[8][3]) {
  for(int model=0;model<MODEL_COUNT;model++) predict(model,input,output[model]);
}

bool validIndices(int index) {
  const SparseModel &model=sparse[index];
  for(int i=0;i<3;i++) {
    const SparseLayer &l=model.layers[i];
    if(l.row[0]!=0 || l.row[l.rows]!=l.nnz || l.stored<(l.nnz?l.nnz:1)) return false;
    for(int r=0;r<l.rows;r++) {
      if(l.row[r]>l.row[r+1] || l.row[r+1]>l.nnz) return false;
      for(uint16_t k=l.row[r];k<l.row[r+1];k++) {
        if(l.col[k]>=l.cols || !std::isfinite(l.values[k])) return false;
        if(k>l.row[r] && l.col[k]<=l.col[k-1]) return false;
      }
    }
  }
  return true;
}

bool selfCheckAll() {
  bool passed=trained_models::selfCheck();
  for(int index=0;index<4;index++) {
    bool ok=validIndices(index);float maxError=0;
    if(ok) for(int sample=0;sample<12;sample++) {
      float output[3];predict(index+4,trained_models::CHECK_INPUTS[sample],output);
      for(int c=0;c<3;c++) {
        float expected=sparse[index].expected[sample][c];
        float error=std::fabs(output[c]-expected);
        if(!std::isfinite(output[c]) || error>1e-5f+1e-5f*std::fabs(expected)) ok=false;
        if(error>maxError) maxError=error;
      }
    }
    Serial.print("CSR H=");Serial.print(trained_models::models[index].hidden);
    Serial.print(" INDEX_AND_OUTPUT_CHECK=");Serial.print(ok?"PASS":"FAIL");
    Serial.print(" MAX_ABS_ERROR=");Serial.println(maxError,8);
    passed=passed&&ok;
  }
  Serial.print("ALL EIGHT MODELS CHECK: ");Serial.println(passed?"PASS":"FAIL");
  return passed;
}
}

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
  if(!printTeacherReferenceResults())return;
  Serial.println("========== FOUR PAIR BENCHMARK START ==========");
  Serial.println("FIRMWARE=TEACHER_RESULT_COMPARE_V10; MODEL_GENERATION=FROZEN_TEACHER_RESULT_V10");
  Serial.println("INPUT_SOURCE=HELDOUT_ARCHIVED_REAL9; NO_NEW_SENSOR_CAPTURE");
  Serial.print("INPUT_SHA256=");Serial.println(comparison_models::BENCHMARK_INPUT_SHA256);
  Serial.println("INPUT_COUNT=9; WARMUP_CALLS=10; CALLS_PER_GROUP=100; GROUPS=5");
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
      comparison_models::predict(i+format*4,comparison_models::BENCHMARK_INPUTS[call%comparison_models::BENCHMARK_INPUT_COUNT],warm);
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

void setup() {
  Serial.begin(9600);delay(1000);
  Serial.println("FIRMWARE=TEACHER_RESULT_COMPARE_V10");
  Serial.println("REFERENCE=result.ppm; TEACHER_REQUIRED_ORIGINAL_FILE");
  if(!comparison_models::selfCheckAll()){Serial.println("MODEL CHECK FAILED");while(true)delay(1000);}
  Serial.println("tを1回送信すると、学習に使っていない保存済み9色のMSEと4組の速度を出します。");
}

void loop() {
  if(Serial.available()) {
    char c=static_cast<char>(Serial.read());
    if(c=='t'||c=='T')runBenchmarks();
    else if(c!='\r'&&c!='\n'&&c!=' '&&c!='\t')Serial.println("半角のtを送信してください。");
  }
  delay(1);
}
