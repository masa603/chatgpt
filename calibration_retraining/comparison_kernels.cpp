namespace comparison_models {
void sparseLayer(const SparseLayer &layer,const float *input,float *output,bool relu) {
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
