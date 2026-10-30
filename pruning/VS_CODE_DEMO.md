# VS Codeで枝刈り・CSR変換を実演する

## ファイルをそろえる

Macに `pruning_demo` などのフォルダを作り、次の3ファイルを入れます。
リンク先の「Download raw file」で保存できます。

- [demo_pruning_csr.py](demo_pruning_csr.py)
- [threshold_tools.py](threshold_tools.py)
- [model_parameters.h](../hdjsk/model_parameters.h)：元のDense用の重み。

リポジトリ全体を取得済みなら、`pruning` フォルダにある実演ファイルをそのまま使えます。
その場合、元の重みは `../hdjsk/model_parameters.h` を自動で見つけます。

## 最初の準備

1. VS Codeの「ファイル → フォルダーを開く」で、上の3ファイルを入れたフォルダを開く。
2. 「ターミナル → 新しいターミナル」を開く。
3. 下のコマンドを順に実行する。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
python demo_pruning_csr.py
```

必要なのはPythonとNumPyです。ArduinoやPyTorchはこの実演には使いません。
すでにこのフォルダの `.venv` を準備済みなら、次回は次の2行で実行できます。

```sh
source .venv/bin/activate
python demo_pruning_csr.py
```

## TAさんの前で見せる順番

`demo_pruning_csr.py` をVS Codeで開き、下のターミナルで実行します。
Enterを押すたびに次の段階へ進みます。

1. **元の重み**：H=40、重み1,840個、Dense配列7,692 byte。第1層の先頭8行を表示。
2. **枝刈り**：実際の `prune()` を呼び、`abs(w) < threshold` を0にする。バイアスは変更しない。
   「0にしただけではDense配列の容量は7,692 byteのまま」という点も表示。
3. **CSRへ変換**：実際の `save_csr_npz()` を呼ぶ。値・列番号・行ポインタを表示し、
   3層全体の配列データ量が1,149 byteになることを示す。
4. **一致を確認**：CSRから重み行列を復元し、枝刈り後DenseとCSRの固定入力に対する推論出力が一致することを表示。

`threshold_tools.py` も開いておくと、`prune()` と `save_csr_npz()` の処理を見せられます。
CSR化で一致を確認する相手は「枝刈り後のDense」です。元のDenseとは枝刈りによって出力が変わり得ます。

`demo_output` に、元の行列・枝刈り後の行列・CSR配列・実行記録・NPZを保存します。
VS Codeの左側から `layer1_before.txt`、`layer1_pruned.txt`、`csr_arrays.txt` を開いて見せることもできます。
既存のモデルや検証結果は上書きしません。

## 実演の条件

デフォルトは既存H=40モデルと暫定閾値 `0.2449173554778099` です。
閾値を指定した後の変換を見せる実演であり、新たな学習や最終採用の判断を行うものではありません。
色精度や校正の制約は [TA向け説明](TA_review.md) に記録しています。

TAさんの前で閾値を変えて試す場合は、例えば次のようにします。

```sh
python demo_pruning_csr.py --threshold 0.01
```

これは手動指定値による実演です。正式な閾値選択は検証MSEと容量を比較して行います。
