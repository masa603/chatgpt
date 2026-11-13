## 実装アルゴリズムとソースコード

PCで学習・閾値選択・枝刈り・CSR変換を行い，Dense重みとCSR配列を数値ヘッダへ出力した．Arduinoでは共通入力から両方式を計算した．重みの形式は両方式ともFP32とした．

CSR変換では各層を行順に走査し，非ゼロ重みを `values`，列番号を `col`，各行の開始位置と終端を `row` へ格納した．列番号は昇順とし，加算順序を元の行列にそろえた．

`trained_models::predict()` はDenseの3層の推論を行う．`comparison_models::sparseLayer()` はCSR形式の1層を計算し，`comparison_models::predict()` は条件番号に応じてDenseまたは3層のCSR推論を呼び出す．隠れ層のみReLUを適用し，中間値を2本の作業配列へ保持する．

新規読取では `normalizeInput()` が黒白校正値による正規化とクリップを行い，`comparison_models::inferAll()` が同一入力から全8条件の出力を求める．`outputTo255()` は出力を0〜255の整数へ変換する．整数RGBをPPMとして保存し，正解画像とのMSEを算出する．

`validIndices()` で行位置の単調性・終端・列範囲を，`selfCheckAll()` で固定入力に対する出力を検査した．CSR変換の正しさは枝刈り後のDense計算と照合し，精度への影響は枝刈り前のDenseとのMSE差で評価した．

学習は `train.py`，候補比較は `select_compression.py`，追加条件による閾値固定は `finalize_compression.py`，ヘッダ出力は `build_program.py` を使用した．受領ログは `record_fresh_capture.py`，単独ビルド表示は `record_single_model_memory.py` で検算・集計した．

CSR推論と計時の関数は，付録の「コメント付きコード」に行ごとの説明を付して掲載する．
