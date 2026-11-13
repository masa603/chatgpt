## 実装アルゴリズムとソースコード

PCで学習・閾値選択・枝刈り・CSR変換を行い，Dense重みとCSR配列を数値ヘッダへ出力した．Arduinoでは共通入力から両方式を計算した．重みの形式は両方式ともFP32とした．

CSR変換では各層を行順に走査し，非ゼロ重みを `values`，列番号を `col` へ格納した．`row` は各行の開始位置と配列終端を示す．列番号は昇順とし，加算順序を元の行列にそろえた．実際の `teacher_compare_v10.ino` の関数を示す．

\Needspace{13\baselineskip}

```cpp
@@CSR_LAYER_SOURCE@@
```

各行はバイアスから開始し，`row[r]` から `row[r+1]` の手前までの非ゼロ重みと，`col[k]` に対応する入力の積を加算する．隠れ層のみReLUを適用する．`comparison_models::predict()` は条件番号に応じてDenseまたは3層のCSR計算を呼び出す．中間値は2本の作業配列に保持する．

新規読取では `normalizeInput()` で正規化・クリップし，`inferAll()` で全8条件を計算する．出力を整数RGBへ変換し，PPMとMSEを出力する．

`validIndices()` で行位置の単調性・終端・列範囲を，`selfCheckAll()` で固定入力に対する出力を検査した．CSR変換の正しさは枝刈り後のDense計算と照合し，精度への影響は枝刈り前のDenseとのMSE差で評価した．

学習は `train.py`，候補比較は `select_compression.py`，追加条件による閾値固定は `finalize_compression.py`，ヘッダ出力は `build_program.py` を使用した．受領ログは `record_fresh_capture.py`，単独ビルド表示は `record_single_model_memory.py` で検算・集計した．
