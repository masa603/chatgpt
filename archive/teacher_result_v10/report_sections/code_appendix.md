## コメント付きコード

実際に使用した `teacher_compare_v10.ino` の関数に，行ごとの説明コメントを加えて示す．

\Needspace{22\baselineskip}

### CSR層の推論

```cpp
void sparseLayer( // CSR形式の1層を計算する関数
  const comparison_models::SparseLayer &layer, // 層の重みと位置情報
  const float *input, // 入力配列を受け取る
  float *output, // 計算結果を書き込む配列
  bool relu // ReLUを適用するかの指定
) { // 関数の処理を開始する
  for (int r = 0; r < layer.rows; r++) { // 出力先の各行を順に計算する
    float value = layer.bias[r]; // その行のバイアスを合計の初期値にする
    for (uint16_t k = layer.row[r]; // その行の非ゼロ重みの開始位置
         k < layer.row[r+1]; // 次の行の開始位置の手前まで繰り返す
         k++) // 次の保存位置へ進む
      value += layer.values[k] * input[layer.col[k]]; // 重みと対応入力の積を加える
    output[r] = (relu && value < 0) ? 0 : value; // 指定時は負の値を0にする
  } // その行の計算を終える
} // 関数を終える
```

\Needspace{27\baselineskip}

### 推論時間の計測

```cpp
uint32_t timeModel( // 100推論の経過時間を返す関数
  int model, // 計時するモデルの番号
  int group, // 計時する組の番号
  double &checksum // 出力の合計を呼出元へ返す参照
) { // 関数の処理を開始する
  float output[3]; // 推論結果のRGBを一時保存する
  checksum = 0; // 出力の合計を初期化する
  uint32_t begin = static_cast<uint32_t>(micros()); // 開始時刻を記録する
  for (int call = 0; call < 100; call++) { // 100回の推論を繰り返す
    int input = (group * 100 + call) // 組を通した呼出番号を求める
      % comparison_models::BENCHMARK_INPUT_COUNT; // 9入力を巡回する番号にする
    comparison_models::predict( // 指定モデルの推論を呼び出す
      model, // モデル番号を渡す
      comparison_models::BENCHMARK_INPUTS[input], // 選んだ保存入力を渡す
      output); // 結果をoutputに受け取る
    checksum += output[0] + output[1] + output[2]; // RGBの出力を合計へ加える
  } // 100回の繰り返しを終える
  return static_cast<uint32_t>(micros()) - begin; // 経過時間をµsで返す
} // 関数を終える
```
