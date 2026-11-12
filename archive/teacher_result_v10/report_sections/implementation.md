## 実装アルゴリズムとソースコード

処理を、PCで行う学習・圧縮と、Arduinoで行う推論・測定に分けた。PC側では、学習した3層の重みから閾値候補を作り、検証用入力で候補を比較する。採用した閾値で重みを0化した後、CSRへ変換し、数値ヘッダへ書き出す。Arduino側ではヘッダを読み込み、同じ入力についてDenseとCSRを計算する。重みの整数化は行わず、両方式ともFP32を使う。

**枝刈りとCSR変換。** 各層の行を出力ニューロン、列を入力ニューロンとする。行を上から順に走査し、枝刈り後に0でない重みだけを `values` へ追加し、その列番号を `columns` へ追加する。`rows[0]=0` とし、1行を処理し終えるたびに、その時点の格納要素数を次の `rows` へ記録する。したがって、行rの非ゼロ要素は `rows[r]` 以上 `rows[r+1]` 未満の範囲になる。列を昇順に走査することで、CSRでも元の行列と同じ順序で積和できる。

\Needspace{14\baselineskip}

説明用に、実測モデルとは別の小さな行列を考える。

$$
W=\begin{pmatrix}0&0.5&0\\-0.2&0&0.1\end{pmatrix}.
$$

$$
\begin{aligned}
\mathrm{values}&=[0.5,-0.2,0.1],\\
\mathrm{columns}&=[1,0,2],\quad\mathrm{rows}=[0,1,3].
\end{aligned}
$$

1行目はvaluesの0番だけ、2行目は1〜2番を使う。行番号を各要素へ持たせず、行の開始位置を共有する点がCSRの特徴である。

**ArduinoのCSR推論。** 次は実際に使用した `teacher_compare_v10.ino` の `comparison_models` 名前空間内にある関数である。ここで `layer.col` はcolumns、`layer.row` はrowsを参照する。

```cpp
@@CSR_LAYER_SOURCE@@
```

各行でバイアスから計算を始め、非ゼロ重みと対応する入力の積だけを足す。隠れ層2層では `relu=true`、出力層では `false` とする。`trained_models::predict()` は全接続を走査するDense計算、`comparison_models::predict()` は条件番号に応じてDenseまたは3層のCSR計算を呼び出す。途中の2層分の出力は固定長配列へ置く。

**入力・出力と検査。** `normalizeInput()` は黒白校正値による正規化と0〜1へのクリップを行う。`comparison_models::inferAll()` は1組の入力から全8条件を計算する。出力を整数RGBへ変換してPPMを出力し、`result.ppm` と同じ色順でMSEを求める。出力色を色番号で置き換える処理は行わない。

`validIndices()` は行位置の単調性、最終行位置と非ゼロ数の一致、列番号の範囲を確認する。`selfCheckAll()` は保存した固定入力について参照出力との誤差を確認する。CSRへの変換が正しいかという検査では、**枝刈り後の重みをDense配列で計算した結果**と比較する。一方、圧縮の精度への影響は、**枝刈り前のDense**と比較する。この2つの比較を区別する。

主なファイルの役割は次のとおりである。数値ヘッダはコードと別ファイルに保存した。

| ファイル | 主な役割 |
| --- | --- |
| `train.py` | 各Hの追加学習、重みと学習条件の保存 |
| `select_compression.py` | 別の検証128入力による閾値候補の比較 |
| `finalize_compression.py` | 学習内9入力の追加条件を確認し、閾値を固定 |
| `build_program.py` | 固定したDense重み・CSR配列と参照値をヘッダへ出力 |
| `teacher_compare_v10.ino` | 保存入力の実機再生、MSE照合、速度測定 |
| `teacher_fresh_v10.ino` | 黒白校正、新規9色の読取、全8条件の出力 |
| `memory_probe_v10.ino` | 指定した1条件だけを含むFlash・静的RAMビルド |
| `record_fresh_capture.py` | 受領ログの入力・出力・MSEの検算と10回の集計 |
| `record_single_model_memory.py` | 条件名付きIDE表示の保存、容量と削減率の集計 |
