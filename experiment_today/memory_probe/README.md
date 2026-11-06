# 各モデル単独のFlash・静的RAMを記録する

精度条件の確認を優先するため、現在は [precision_followup](../precision_followup/README.md) の実機確認へ進みます。
このフォルダのCSRは元の候補です。追加候補のメモリ使用量と混同しないでください。
40 DenseのFlash 60,268 byte・静的RAM 6,764 byteは受領し、[single_model_memory.csv](../single_model_memory.csv) に保存しています。

測定済みの速度・紙の再読取の後に、各モデルを1つずつ含むプログラムのビルド使用量を比較します。
全6モデル入りの `fresh_scan_all` の133,424 byte・7,268 byteとは、スケッチと用途を分けて記録します。

1. `memory_probe` フォルダへ [memory_probe.ino](memory_probe.ino) と [model_data.h](model_data.h) を保存します。
2. Arduino IDEで開き、**Arduino UNO R4 WiFi**を選びます。
3. 最初はファイル上部の設定をそのまま使います。**H=40・Dense**です。
4. **左上の✓（検証）**を押します。書き込みやセンサの読み取りは不要です。
5. 完了後、下の出力の「スケッチが…バイト」「グローバル変数が…バイト」の2行を保存します。
   条件名「40 Dense」も一緒に残します。
6. 下表に従って上部の2値を変更し、残り5条件も同じ操作を行います。

| 条件 | `PROBE_H` | `PROBE_CSR` |
| --- | ---: | ---: |
| 40 Dense | 40 | 0 |
| 40 CSR | 40 | 1 |
| 60 Dense | 60 | 0 |
| 60 CSR | 60 | 1 |
| 100 Dense | 100 | 0 |
| 100 CSR | 100 | 1 |

設定するのは次の2行です。各条件でボード・コア・最適化設定は統一します。

```cpp
#define PROBE_H 40
#define PROBE_CSR 0
```

このプログラムでは、プリプロセッサの条件分岐により各ビルドへ1モデル・1形式だけを含めます。
元のベンチマークと同じ推論関数、同じ凍結済み重み・閾値を使います。
Dense・CSRともに配列はconst FP32、バイアスは同一です。
入力・出力のラッパーは共通です。volatile入力と出力によりモデル全体の計算が残るようにしています。
比較用の128入力や正解画像、他のモデル、MSE計算、センサ読取は含みません。

Flashの値はモデル配列だけでなく、推論コード・共通ラッパー・Arduinoコア等を含みます。
RAMはIDEが表示するグローバル変数等の静的領域です。
隠れ層の作業配列はスタックを使うので、この値をピークRAMや推論時の全RAMと呼ばないでください。
const配列のため、CSR化で静的RAMがほぼ変わらずFlashが減る可能性があります。そのまま結果に残します。

結果の空欄テンプレートは [single_model_memory_template.csv](../single_model_memory_template.csv) です。
PC検証記録は [memory_probe_verification.json](../memory_probe_verification.json) に保存します。
PCの実行ファイルサイズをArduinoのFlash・RAMとして記録する処理はありません。
