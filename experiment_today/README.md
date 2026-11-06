# 2026年11月6日：40→60→100次元の比較

実機ログの受領状況は [hardware](hardware/README.md)、実測値の集計は
[device_results.csv](device_results.csv) に保存しています。40次元の計算確認・速度測定はPASSです。

3時間の作業では、学習済みのモデルを使って**40→60→100**の順に実機ログを残します。
学習はすでに完了しています。新しい33組で学習した重みから、各Hの閾値を選び直しました。
旧H=40の閾値0.2449と旧CSRモデルは使いません。

## 最初に40次元

1. `compare_h40` フォルダへ [compare_h40.ino](compare_h40/compare_h40.ino) と
   [parameters.h](compare_h40/parameters.h) を保存します。
2. Arduino IDEで `.ino` を開き、**Arduino UNO R4 WiFi** と接続ポートを選んで書き込みます。
3. コンパイル時に表示するFlash・RAM使用量を保存します。
   このスケッチにはDenseとCSRの両方が入っているため、表示値は**比較プログラム全体**の使用量です。
4. シリアルモニタを **9600 bps** にし、必要ならRESETを押します。
5. 半角の **`t`** を送信します。画像サイズやb・wの入力はありません。
6. **`BENCHMARK START`〜`BENCHMARK END`** を丸ごと保存します。

`CSR INDEX CHECK: PASS`、`FIXED INPUT CHECK: Dense=PASS CSR=PASS`、
`TIMING VALID: YES` を確認します。PASSの確認前に次のモデルへ進めません。
エラーやFAILが出た場合は、その表示を保存します。

## 次に60、最後に100

同じ操作を以下のフォルダで繰り返します。各フォルダの2ファイルを組にして使います。

| 順序 | スケッチ | 同じフォルダに置くヘッダ |
| --- | --- | --- |
| 1 | [compare_h40.ino](compare_h40/compare_h40.ino) | [parameters.h](compare_h40/parameters.h) |
| 2 | [compare_h60.ino](compare_h60/compare_h60.ino) | [parameters.h](compare_h60/parameters.h) |
| 3 | [compare_h100.ino](compare_h100/compare_h100.ino) | [parameters.h](compare_h100/parameters.h) |

H=100のDense配列は43,212 byteですが、全モデル配列はconstです。
配列のbyte数と、Arduino IDEが報告する実際のRAM・Flash使用量を分けて記録します。
比較スケッチのビルド成功だけでは、各方式単独の最小RAM・Flash使用量を測ったことにはなりません。

## 1回のt送信で記録するもの

- 新しいモデルのH・元ヘッダのSHA256・枝刈り閾値。
- Dense/CSRそれぞれのモデル配列容量。
- **保存した学習入力**でのMSE：3×3、6×4、全33組。
- **保存した検証入力**でのMSE：独立に読み取った2回分とその平均。
- 実機での推論時間：10回ウォームアップ後、100推論を5組。
  Dense先行・CSR先行を交互にし、1推論当たりの平均と標本標準偏差を表示。

センサの読取・待機・シリアル出力・MSE計算は推論時間に含めません。
計測後にチェックサムを表示し、結果が使われない演算の削除を防ぎます。
固定入力128件の数値確認を通過した後に計時します。

## 今回のPC結果

同じ保存入力へFP32推論し、0〜1クリップ、255倍、四捨五入してMSEを求めました。
全33組の入力を残しており、都合のよい色だけを除く処理はありません。

| H | 閾値 | Dense byte | CSR byte | 学習33組MSE Dense→CSR | 検証平均MSE Dense→CSR | 検証変化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 0.1387345567 | 7,692 | 1,109 | 70.03→590.55 | 4,499.22→4,634.67 | +3.01% |
| 60 | 0.1484270990 | 16,332 | 1,279 | 70.01→263.28 | 2,065.91→2,142.42 | +3.70% |
| 100 | 0.2595954686 | 43,212 | 1,739 | 70.01→1,144.07 | 3,119.15→3,258.82 | +4.48% |

閾値は重みの絶対値から作る全候補を評価し、検証平均MSEの増加5%以下かつDense未満の容量を条件に、
容量が最小になるものを選びました。**学習MSEの増加5%以下を条件にした選択ではありません。**
表のように学習MSEは悪化しているため、全データで精度を維持したとは報告しません。
検証2回目には異なる6色の入力がすべて0になった箇所があるため、候補は暫定扱いのままです。
検証結果を、絶対的な色精度が十分という判定には使いません。

正解の学習RGBは元Notebookの保存済み表示から復元した値で、元PNGとの数値照合は未実施です。
出典は [retraining/data/provenance.json](../retraining/data/provenance.json) にあります。
各候補・重みのSHA256・検証データのSHA256は `selection/h40` などへ保存しています。
[PC結果のCSV](pc_results.csv) と [実機記録の空欄テンプレート](device_results_template.csv) も用意しています。

## 最終評価の扱い

利用者の申告では、別の最終評価チャートは手元になく、当日の印刷もできません。
そのため、今回は保存済みの学習・検証入力をArduino上で再生する比較までを記録します。
**独立した最終評価は未実施**であり、閾値を選んだ検証チャートを最終評価と呼び替えません。
この手順では新たな画像読取や、当初計画の各条件10回の物理測定も行っていません。
速度の標準偏差5組と、物理的な再読取のばらつきは別です。
変更点は [protocol_change.json](protocol_change.json) に記録しました。

## 再生成と確認

リポジトリ直下から、NumPyを導入したPythonで実行できます。

```sh
python3 experiment_today/select_models.py
python3 experiment_today/export_experiments.py
python3 experiment_today/verify_host.py
```

保存済みの選択結果は、元重みと検証CSVのSHA256が同じ場合に再利用します。
ホストC++11で全3モデルをコンパイルし、各417入力でDense/CSRのRGB8がPythonと一致し、
シリアルt、MSE各行、索引検査、標本標準偏差の計算を確認しました（ASan・UBSan有効）。
[verification.json](verification.json) はPC上の確認記録です。
PC上の時間をArduinoの時間として保存する処理はありません。実機結果はログ受領後に追記します。
