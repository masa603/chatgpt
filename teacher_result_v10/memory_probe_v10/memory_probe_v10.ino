// Change only these two values for an Arduino IDE Verify build.
#ifndef PROBE_H
#define PROBE_H 40
#endif
#ifndef PROBE_CSR
#define PROBE_CSR 0  // 0=Dense,1=pruning+CSR
#endif

#include <Arduino.h>
#include "memory_model_v10.h"

#if defined(__GNUC__)
#define PROBE_NOINLINE __attribute__((noinline,noclone))
#else
#define PROBE_NOINLINE
#endif

void modelPredict(const float *input,float *output);
volatile float probeInput[3]={0.2f,0.4f,0.6f};
volatile float probeOutput[3]={0,0,0};

#if PROBE_CSR == 0
PROBE_NOINLINE void modelPredict(const float *input,float *output) {
  const memory_model::DenseModel &m=memory_model::MODEL;
  float layer1[100],layer2[100];
  for(int r=0;r<m.hidden;r++) {
    float value=m.b1[r];
    for(int c=0;c<3;c++)value+=m.w1[r*3+c]*input[c];
    layer1[r]=value>0.0f?value:0.0f;
  }
  for(int r=0;r<m.hidden;r++) {
    float value=m.b2[r];
    for(int c=0;c<m.hidden;c++)value+=m.w2[r*m.hidden+c]*layer1[c];
    layer2[r]=value>0.0f?value:0.0f;
  }
  for(int r=0;r<3;r++) {
    float value=m.b3[r];
    for(int c=0;c<m.hidden;c++)value+=m.w3[r*m.hidden+c]*layer2[c];
    output[r]=value;
  }
}
#else
void sparseLayer(const memory_model::SparseLayer &layer,const float *input,float *output,bool relu) {
  for(int r=0;r<layer.rows;r++) {
    float value=layer.bias[r];
    for(uint16_t k=layer.row[r];k<layer.row[r+1];k++)value+=layer.values[k]*input[layer.col[k]];
    output[r]=(relu&&value<0)?0:value;
  }
}
PROBE_NOINLINE void modelPredict(const float *input,float *output) {
  const memory_model::SparseModel &m=memory_model::MODEL;
  float first[100],second[100];
  sparseLayer(m.layers[0],input,first,true);
  sparseLayer(m.layers[1],first,second,true);
  sparseLayer(m.layers[2],second,output,false);
}
#endif

void setup() {
  Serial.begin(9600);delay(1000);
  Serial.println("FIRMWARE=V10_SINGLE_MODEL_MEMORY");
  Serial.print("H=");Serial.println(PROBE_H);
  Serial.print("FORMAT_ID=");Serial.println(PROBE_CSR);
  Serial.print("MODEL_SOURCE_SHA256=");Serial.println(memory_model::MODEL_SOURCE_SHA256);
  Serial.print("THRESHOLD=");Serial.println(memory_model::THRESHOLD,17);
  Serial.print("MODEL_ARRAY_BYTES=");Serial.println(memory_model::MODEL_ARRAY_BYTES);
  Serial.println("SCOPE=SINGLE_MODEL_BUILD; STATIC_RAM_ONLY; STACK_PEAK_NOT_MEASURED");
}

void loop() {
  float input[3]={probeInput[0],probeInput[1],probeInput[2]},output[3];
  modelPredict(input,output);
  for(int c=0;c<3;c++)probeOutput[c]=output[c];
  delay(100);
}
