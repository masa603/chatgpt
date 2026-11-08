"""One t replays the received nine colors, then times all four frozen Dense/CSR pairs."""

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'pruning'))
from export_csr_arduino import array_definition,dense_layer_references
from threshold_tools import load_model


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    source=HERE/'physical_readings/capture_01'
    summary=json.loads((source/'summary.json').read_text())
    assert summary['firmware']=='COLOR_RETRAIN_V9' and summary['training_role']=='NOT_USED_FOR_V9_FITTING_OR_DEVELOPMENT'
    with (source/'inputs.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    raw=np.array([[r[f'raw_{c}'] for c in 'rgb'] for r in rows],np.float32)
    black=np.array(summary['black_rgb'],np.float32);white=np.array(summary['white_rgb'],np.float32)
    inputs=np.clip((raw-black)/(white-black),0,1)
    original=[]
    for h in [40,60,80,100]:
        ppm=(source/f'h{h}_dense.ppm').read_text().split()
        pixels=np.array(ppm[4:],np.uint8).reshape(9,3)
        layers=load_model(HERE/f'models/h{h}/model_parameters.h')
        computed=np.floor(np.clip(dense_layer_references(layers,inputs)[-1].astype(float),0,1)*255+.5).astype(np.uint8)
        assert np.array_equal(pixels,computed)
        original.append(pixels)
    output=HERE/'compare_saved_v9';output.mkdir(exist_ok=True)
    header=(HERE/'compare_all_v9/comparison_parameters_v9.h').read_text()
    header+='\nnamespace saved_capture {\n'
    header+=f'const char SOURCE_LOG_SHA256[]="{sha(source/"submitted_log.txt")}";\n'
    header+=f'const char INPUT_CSV_SHA256[]="{sha(source/"inputs.csv")}";\n'
    header+=array_definition('INPUTS',inputs,'float')
    header+=array_definition('ORIGINAL_DENSE_RGB',np.array(original,np.uint8).reshape(36,3),'uint8_t')
    pure_reference=ROOT/'learning_followup/reference_audit/provided/result.ppm'
    tokens=pure_reference.read_text().split();assert tokens[:4]==['P3','3','3','255'] and len(tokens)==31
    header+=array_definition('SITE_RESULT_PURE_RGB',np.array(tokens[4:],np.uint8).reshape(9,3),'uint8_t')
    header+=f'const char SITE_RESULT_REFERENCE_SHA256[]="{sha(pure_reference)}";\n'+'}\n'
    (output/'saved_parameters_v9.h').write_text(header)
    dense=(HERE/'color_retrain_v9/color_retrain_v9.ino').read_text()
    begin=dense.index('namespace trained_models {')
    end=dense.index('\nvoid outputNNInputs();',begin)
    dense_code=dense[begin:end]
    sketch='#include <Arduino.h>\n#include <cmath>\n#include "saved_parameters_v9.h"\n\n'+dense_code+'\n'
    sketch+=(HERE/'comparison_kernels.cpp').read_text()+'\n'+(HERE/'saved_replay.cpp').read_text()+'\n'
    benchmark=(HERE/'comparison_benchmark.cpp').read_text().replace('COLOR_COMPARE_V9','COLOR_SAVED_COMPARE_V9')
    marker='  Serial.println("========== FOUR PAIR BENCHMARK START ==========");'
    benchmark=benchmark.replace(marker,'  if(!replaySavedCapture())return;\n'+marker)
    sketch+=benchmark
    sketch+='''
void setup() {
  Serial.begin(9600);delay(1000);
  Serial.println("FIRMWARE=COLOR_SAVED_COMPARE_V9");
  if(!comparison_models::selfCheckAll()) {Serial.println("MODEL CHECK FAILED");while(true)delay(1000);}
  Serial.println("準備完了。tを1回送信すると保存済み9色のDense/CSR精度と、4組の速度・容量を表示します。");
  Serial.println("紙の読取や黒白の再登録は行いません。");
}

void loop() {
  if(Serial.available()) {
    char command=static_cast<char>(Serial.read());
    if(command=='t' || command=='T')runBenchmarks();
    else if(command!='\\r' && command!='\\n' && command!=' ' && command!='\\t')Serial.println("半角のtを送信してください。");
  }
  delay(1);
}
'''
    (output/'compare_saved_v9.ino').write_text(sketch)
    report={'input_source':'Archived physical COLOR_RETRAIN_V9 capture1; no new sensor reading.',
            'source_log_sha256':sha(source/'submitted_log.txt'),'inputs_csv_sha256':sha(source/'inputs.csv'),
            'input_count':9,'original_dense_rgb_exactly_reproduced':True,
            'model_or_threshold_changed':False,'code_sha256':sha(output/'compare_saved_v9.ino'),
            'parameters_sha256':sha(output/'saved_parameters_v9.h'),
            'hardware_saved_replay':'PENDING','hardware_benchmark':'PENDING'}
    (HERE/'saved_comparison_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Built paper-free saved-input comparison: t outputs eight MSEs/PPMs then all four speed comparisons. Frozen weights/thresholds unchanged.')


if __name__=='__main__':build()
