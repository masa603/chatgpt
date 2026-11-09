"""Build a real ten-chart-reading scanner around the already frozen V10 models."""

from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiment_today'))
from build_fresh_scanner import replace_function


SERIAL=r'''void promptFreshPatch() {
  const char *names[9]={"赤","緑","青","黄","水色","紫","黒","灰","白"};
  Serial.print("測定回 ");Serial.print(captureNumber);
  Serial.print("：次は ");Serial.print(names[currentRow*3+currentColumn]);
  Serial.print("（");Serial.print(currentRow+1);Serial.print("行");Serial.print(currentColumn+1);
  Serial.println("列）を当てて r を1回送信してください。");
}

void handleSerialCommand() {
  if(Serial.available()<=0)return;
  int command=Serial.read();
  switch(command) {
    case 'n':case 'N':
      if(!imageFinished && (calibrationStep>0 || currentRow>0 || currentColumn>0)) {
        Serial.println("========== INCOMPLETE CAPTURE ABORTED ==========");
        Serial.print("CAPTURE=");Serial.println(captureNumber);
        Serial.print("COLORS_READ=");Serial.println(currentRow*3+currentColumn);
        Serial.println("REASON=USER_RESTARTED_INCOMPLETE_READ; NOT_COUNTED_AS_COMPLETED");
      }
      calibrationStep=0;currentRow=0;currentColumn=0;imageFinished=false;captureNumber++;
      Serial.print("測定回 ");Serial.print(captureNumber);
      Serial.println("：黒 b → 白 w → 9色を各 r で読み取ってください。");
      break;
    case 'b':case 'B':
      if(calibrationStep!=0){Serial.println("黒は登録済みです。白 w、色 r、次の回 n を送信してください。");return;}
      handleCalibration();break;
    case 'w':case 'W':
      if(calibrationStep!=1){Serial.println("先に黒を当てて b を送信してください。");return;}
      handleCalibration();if(calibrationStep==2)promptFreshPatch();break;
    case 'r':case 'R':
      handleRead();if(calibrationStep==2 && !imageFinished)promptFreshPatch();break;
    case 't':case 'T':
      Serial.println("この版は紙の新規読取用です。黒 b、白 w、各色 r、次の回 n を使います。");break;
    case '\r':case '\n':case ' ':case '\t':break;
    default:Serial.println("半角で b（黒）、w（白）、r（色）、n（次の回）を送信してください。");
  }
}'''


def build():
    folder=HERE/'teacher_fresh_v10';folder.mkdir(exist_ok=True)
    frozen=HERE/'teacher_compare_v10'
    (folder/'teacher_parameters_v10.h').write_bytes((frozen/'teacher_parameters_v10.h').read_bytes())
    previous=(ROOT/'calibration_retraining/compare_all_v9/compare_all_v9.ino').read_text()
    current=(frozen/'teacher_compare_v10.ino').read_text()
    # Exactly the same dense/CSR kernels and numeric model header as the verified t version.
    prefix=current[:current.index('bool printTeacherReferenceResults()')]
    code=prefix+previous[previous.index('void outputNNInputs();'):]
    code=code.replace('unsigned int captureNumber=1;','unsigned int captureNumber=1;\nunsigned int completedCaptures=0;')
    code=code.replace('FIRMWARE=COLOR_COMPARE_V9','FIRMWARE=TEACHER_FRESH_V10')
    code=code.replace('V9のDense/CSR・全8条件準備完了','V10の10回新規読取・全8条件準備完了')
    code=code.replace('DISTRIBUTED_NOTEBOOK_CMYK_DISPLAY_RECOVERED','TEACHER_REQUIRED_ORIGINAL_RESULT_PPM')
    code=code.replace('MODEL_GENERATION=FROZEN_V9','MODEL_GENERATION=FROZEN_V10')
    code=code.replace('PAIRED CAPTURE START','TEACHER FRESH CAPTURE START').replace('PAIRED CAPTURE END','TEACHER FRESH CAPTURE END')
    code=code.replace('  Serial.print("CAPTURE=");Serial.println(captureNumber);',
        '  completedCaptures++;\n  Serial.print("CAPTURE=");Serial.println(captureNumber);\n  Serial.print("COMPLETED_CAPTURES_THIS_BOOT=");Serial.println(completedCaptures);\n  Serial.println("TARGET_PHYSICAL_CAPTURES=10; MODELS_AND_THRESHOLDS_FROZEN");',1)
    code=code.replace('  Serial.println("INDEPENDENT_FINAL_EVALUATION=NOT_PERFORMED");',
        '  Serial.println("INDEPENDENT_FINAL_EVALUATION=NOT_PERFORMED");\n  Serial.println("SAMPLE_COUNT=10; RGB_FLASH_DELAY_MS=10; LED_DRIVE=ANALOG_WRITE_255_0");',1)
    code=code.replace('  Serial.println("速度・容量比較は t、次の9色読取は n を送信してください。");',
        '  Serial.print("完了した9色読取：");Serial.print(completedCaptures);Serial.println(" / 10回");\n'
        '  if(completedCaptures<10)Serial.println("この回のログを保存してから n を送信し、黒白校正から次の回を読み取ってください。");\n'
        '  else Serial.println("10回分以上の9色読取が完了しました。全回のログを保存してください。");')
    # Existing speed data remain in their own verified record. t never replays old inputs here.
    begin=code.index('#if defined(__GNUC__)',code.index('// シリアルモニタから操作'))
    end=code.index('void handleSerialCommand()',begin)
    code=code[:begin]+code[end:]
    code=replace_function(code,'handleSerialCommand',SERIAL)
    code=code.replace('③ 各色を当てて r を1回ずつ送信。速度・容量比較は t',
        '③ 赤→緑→青→黄→水色→紫→黒→灰→白の順で各 r。9色後にログ保存、次は n')
    assert 'CMYK' not in code and 'FROZEN_V9' not in code and 'runBenchmarks' not in code
    assert 'BENCHMARK_INPUTS[' not in code
    (folder/'teacher_fresh_v10.ino').write_text(code)
    print('Built real-reading V10 scanner; exact frozen header/kernel, all8 models per new9-color capture, completed-reading counter, incomplete-abort record.')


if __name__=='__main__':build()
