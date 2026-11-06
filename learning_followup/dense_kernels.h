namespace trained_models {
struct Model {
  int hidden;
  const float *w1, *b1, *w2, *b2, *w3, *b3;
  const char *sha256;
  const float (*expected)[3];
};
// INSERT_DEFINITIONS

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

void inferAll(const float *input,float output[3][3]) {
  for(int i=0;i<3;i++) predict(i,input,output[i]);
}

bool selfCheck() {
  bool passed=true;
  Serial.println("========== RETRAIN CHECK START ==========");
  Serial.print("TRAINING_DATA_SHA256="); Serial.println(DATA_SHA256);
  Serial.print("TRAINING_ROWS="); Serial.println(TRAINING_ROWS);
  Serial.println("FORMAT=DENSE_FP32_ONLY");
  for(int i=0;i<3;i++) {
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
