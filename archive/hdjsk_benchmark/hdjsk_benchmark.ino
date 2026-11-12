#include <Arduino.h>
#include <cmath>
#include "benchmark_parameters.h"

#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#define COMPILER_BARRIER() __asm__ __volatile__("" ::: "memory")
#else
#define NOINLINE
#define COMPILER_BARRIER()
#endif

// Explicit declarations keep Arduino's prototype generation independent of attributes.
void densePredict(const float *input, float *output);
void csrPredict(const float *input, float *output);
uint32_t timeBatch(void (*predict)(const float *, float *), uint16_t firstInput, double &checksum);

constexpr uint16_t WARMUP_CALLS = 10;
constexpr uint16_t CALLS_PER_GROUP = 100;
constexpr uint16_t GROUP_COUNT = 5;
static_assert(GROUP_COUNT > 1, "Sample standard deviation needs multiple groups.");

// Outputs remain observable after the timer; both kernels write the same buffer.
float benchmarkOutputs[CALLS_PER_GROUP][3];
volatile double checksumSink = 0.0;

NOINLINE void densePredict(const float *input, float *output) {
  float layer1[CSR_HIDDEN_SIZE] = {0};
  float layer2[CSR_HIDDEN_SIZE] = {0};
  for (int row = 0; row < CSR_HIDDEN_SIZE; row++) {
    float sum = bias_1[row];
    for (int col = 0; col < 3; col++) sum += dense_weight_1[row * 3 + col] * input[col];
    layer1[row] = sum > 0.0f ? sum : 0.0f;
  }
  for (int row = 0; row < CSR_HIDDEN_SIZE; row++) {
    float sum = bias_2[row];
    for (int col = 0; col < CSR_HIDDEN_SIZE; col++) sum += dense_weight_2[row * CSR_HIDDEN_SIZE + col] * layer1[col];
    layer2[row] = sum > 0.0f ? sum : 0.0f;
  }
  for (int row = 0; row < 3; row++) {
    float sum = bias_3[row];
    for (int col = 0; col < CSR_HIDDEN_SIZE; col++) sum += dense_weight_3[row * CSR_HIDDEN_SIZE + col] * layer2[col];
    output[row] = sum;
  }
}

void sparseLayer(const float *input, const float *values, const uint8_t *columns,
                 const uint16_t *pointers, const float *bias, int rows,
                 float *output, bool useRelu) {
  for (int row = 0; row < rows; row++) {
    float sum = bias[row];
    for (uint16_t p = pointers[row]; p < pointers[row + 1]; p++) sum += values[p] * input[columns[p]];
    output[row] = useRelu ? (sum > 0.0f ? sum : 0.0f) : sum;
  }
}

NOINLINE void csrPredict(const float *input, float *output) {
  float layer1[CSR_HIDDEN_SIZE] = {0};
  float layer2[CSR_HIDDEN_SIZE] = {0};
  sparseLayer(input, weight_1_values, weight_1_col_idx, weight_1_row_ptr, bias_1, CSR_HIDDEN_SIZE, layer1, true);
  sparseLayer(layer1, weight_2_values, weight_2_col_idx, weight_2_row_ptr, bias_2, CSR_HIDDEN_SIZE, layer2, true);
  sparseLayer(layer2, weight_3_values, weight_3_col_idx, weight_3_row_ptr, bias_3, 3, output, false);
}

size_t denseModelBytes() {
  return sizeof(dense_weight_1) + sizeof(dense_weight_2) + sizeof(dense_weight_3)
       + sizeof(bias_1) + sizeof(bias_2) + sizeof(bias_3);
}

size_t csrModelBytes() {
  return sizeof(weight_1_values) + sizeof(weight_1_col_idx) + sizeof(weight_1_row_ptr) + sizeof(bias_1)
       + sizeof(weight_2_values) + sizeof(weight_2_col_idx) + sizeof(weight_2_row_ptr) + sizeof(bias_2)
       + sizeof(weight_3_values) + sizeof(weight_3_col_idx) + sizeof(weight_3_row_ptr) + sizeof(bias_3);
}

bool checkOutput(const float *actual, const float *expected, float &maxError) {
  bool passed = true;
  for (int c = 0; c < 3; c++) {
    if (!std::isfinite(actual[c]) || !std::isfinite(expected[c])) {
      passed = false;
      continue;
    }
    float error = std::fabs(actual[c] - expected[c]);
    if (error > maxError) maxError = error;
    if (error > 1.0e-5f + 1.0e-5f * std::fabs(expected[c])) passed = false;
  }
  return passed;
}

void fixedInputCheck(bool &densePassed, bool &csrPassed, float &denseError, float &csrError) {
  densePassed = csrPassed = true;
  denseError = csrError = 0.0f;
  for (uint16_t i = 0; i < BENCHMARK_INPUT_COUNT; i++) {
    float denseOutput[3], csrOutput[3];
    densePredict(BENCHMARK_INPUTS[i], denseOutput);
    csrPredict(BENCHMARK_INPUTS[i], csrOutput);
    if (!checkOutput(denseOutput, DENSE_EXPECTED_OUTPUTS[i], denseError)) densePassed = false;
    if (!checkOutput(csrOutput, CSR_EXPECTED_OUTPUTS[i], csrError)) csrPassed = false;
  }
}

double outputChecksum(uint16_t count) {
  double checksum = 0.0;
  for (uint16_t i = 0; i < count; i++) {
    for (int c = 0; c < 3; c++) checksum += benchmarkOutputs[i][c];
  }
  checksumSink = checksum;
  return checksum;
}

NOINLINE uint32_t timeBatch(void (*predict)(const float *, float *), uint16_t firstInput, double &checksum) {
  COMPILER_BARRIER();
  uint32_t start = static_cast<uint32_t>(micros());
  COMPILER_BARRIER();
  for (uint16_t i = 0; i < CALLS_PER_GROUP; i++) {
    uint16_t index = (firstInput + i) % BENCHMARK_INPUT_COUNT;
    predict(BENCHMARK_INPUTS[index], benchmarkOutputs[i]);
  }
  COMPILER_BARRIER();
  uint32_t elapsed = static_cast<uint32_t>(micros()) - start;
  COMPILER_BARRIER();
  checksum = outputChecksum(CALLS_PER_GROUP);
  return elapsed;
}

void timingStatistics(const uint32_t *totals, double &mean, double &stddev) {
  mean = 0.0;
  for (uint16_t i = 0; i < GROUP_COUNT; i++) mean += static_cast<double>(totals[i]) / CALLS_PER_GROUP;
  mean /= GROUP_COUNT;
  double squared = 0.0;
  for (uint16_t i = 0; i < GROUP_COUNT; i++) {
    double difference = static_cast<double>(totals[i]) / CALLS_PER_GROUP - mean;
    squared += difference * difference;
  }
  stddev = std::sqrt(squared / (GROUP_COUNT - 1));
}

void printMetadata() {
  Serial.println("========== BENCHMARK START ==========");
  Serial.print("MODEL_STATUS="); Serial.println(CSR_PROVISIONAL ? "PROVISIONAL" : "SELECTED");
  Serial.print("H="); Serial.println(CSR_HIDDEN_SIZE);
  Serial.print("THRESHOLD="); Serial.println(CSR_THRESHOLD_TEXT);
  Serial.print("MODEL_SOURCE_SHA256="); Serial.println(CSR_SOURCE_MODEL_SHA256);
  Serial.print("INPUT_SHA256="); Serial.println(BENCHMARK_INPUT_SHA256);
  Serial.print("INPUT_COUNT="); Serial.println(BENCHMARK_INPUT_COUNT);
  Serial.print("WARMUP_CALLS="); Serial.println(WARMUP_CALLS);
  Serial.print("CALLS_PER_GROUP="); Serial.println(CALLS_PER_GROUP);
  Serial.print("GROUPS="); Serial.println(GROUP_COUNT);
  Serial.print("COMPILER="); Serial.println(__VERSION__);
#ifdef F_CPU
  Serial.print("F_CPU_HZ="); Serial.println(F_CPU);
#endif
  Serial.print("MODEL BYTES: Dense="); Serial.print(denseModelBytes());
  Serial.print(" CSR="); Serial.println(csrModelBytes());
}

void runComparison() {
  bool densePassed, csrPassed;
  float denseError, csrError;
  fixedInputCheck(densePassed, csrPassed, denseError, csrError);
  bool bytesMatch = denseModelBytes() == CSR_DENSE_REFERENCE_BYTES && csrModelBytes() == CSR_EXPECTED_MODEL_BYTES;
  if (!densePassed || !csrPassed || !bytesMatch) {
    printMetadata();
    Serial.print("FIXED INPUT CHECK: Dense="); Serial.print(densePassed ? "PASS" : "FAIL");
    Serial.print(" CSR="); Serial.println(csrPassed ? "PASS" : "FAIL");
    Serial.println("TIMING NOT RUN: check outputs and model byte counts.");
    Serial.println("=========== BENCHMARK END ===========");
    return;
  }
  // Explicit warmup after correctness checks; every output is consumed.
  for (uint16_t i = 0; i < WARMUP_CALLS; i++) {
    densePredict(BENCHMARK_INPUTS[i % BENCHMARK_INPUT_COUNT], benchmarkOutputs[0]);
    outputChecksum(1);
    csrPredict(BENCHMARK_INPUTS[i % BENCHMARK_INPUT_COUNT], benchmarkOutputs[0]);
    outputChecksum(1);
  }
  uint32_t denseTotals[GROUP_COUNT], csrTotals[GROUP_COUNT];
  double denseChecksums[GROUP_COUNT], csrChecksums[GROUP_COUNT];
  for (uint16_t group = 0; group < GROUP_COUNT; group++) {
    uint16_t first = (group * CALLS_PER_GROUP) % BENCHMARK_INPUT_COUNT;
    if (group % 2 == 0) {
      denseTotals[group] = timeBatch(densePredict, first, denseChecksums[group]);
      csrTotals[group] = timeBatch(csrPredict, first, csrChecksums[group]);
    } else {
      csrTotals[group] = timeBatch(csrPredict, first, csrChecksums[group]);
      denseTotals[group] = timeBatch(densePredict, first, denseChecksums[group]);
    }
  }
  double denseMean, denseStddev, csrMean, csrStddev;
  timingStatistics(denseTotals, denseMean, denseStddev);
  timingStatistics(csrTotals, csrMean, csrStddev);
  // All serial output and checksums are outside the timed interval.
  printMetadata();
  Serial.println("FIXED INPUT CHECK: Dense=PASS CSR=PASS");
  Serial.print("MAX ABS ERROR: Dense="); Serial.print(denseError, 8);
  Serial.print(" CSR="); Serial.println(csrError, 8);
  Serial.println("group,first_model,dense_total_us,csr_total_us,dense_us_per_inference,csr_us_per_inference,dense_checksum,csr_checksum");
  bool nonzeroTimes = true;
  for (uint16_t group = 0; group < GROUP_COUNT; group++) {
    if (denseTotals[group] == 0 || csrTotals[group] == 0) nonzeroTimes = false;
    Serial.print(group + 1); Serial.print(","); Serial.print(group % 2 == 0 ? "Dense" : "CSR");
    Serial.print(","); Serial.print(denseTotals[group]); Serial.print(","); Serial.print(csrTotals[group]);
    Serial.print(","); Serial.print(static_cast<double>(denseTotals[group]) / CALLS_PER_GROUP, 4);
    Serial.print(","); Serial.print(static_cast<double>(csrTotals[group]) / CALLS_PER_GROUP, 4);
    Serial.print(","); Serial.print(denseChecksums[group], 8); Serial.print(","); Serial.println(csrChecksums[group], 8);
  }
  Serial.print("DENSE mean_us="); Serial.print(denseMean, 4); Serial.print(" sample_stddev_us="); Serial.println(denseStddev, 4);
  Serial.print("CSR mean_us="); Serial.print(csrMean, 4); Serial.print(" sample_stddev_us="); Serial.println(csrStddev, 4);
  if (denseMean > 0.0) {
    Serial.print("CSR time_change_percent="); Serial.println((csrMean / denseMean - 1.0) * 100.0, 4);
  }
  Serial.print("TIMING VALID: "); Serial.println(nonzeroTimes ? "YES" : "NO (timer resolution)");
  Serial.println("SCOPE: inference on saved inputs; provisional model; color accuracy assessed separately.");
  Serial.println("=========== BENCHMARK END ===========");
}

void setup() {
  Serial.begin(9600);
  delay(1000);
  Serial.println("Dense/CSR speed comparison: send t to start.");
}

void loop() {
  if (Serial.available() > 0) {
    int command = Serial.read();
    if (command == 't' || command == 'T') runComparison();
    else if (command != '\r' && command != '\n' && command != ' ' && command != '\t') {
      Serial.println("Send t to start the comparison.");
    }
  }
}
