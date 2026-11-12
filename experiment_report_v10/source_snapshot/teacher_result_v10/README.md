# 先生指定のresult.ppmへ学習を修正したV10

**2026年11月12日受領：各モデル単独のFlash・静的RAM、全8条件を記録しました。** [結果表・元のIDE表示](hardware/single_model_memory_run_01/README.md)、[8条件CSV](single_model_memory_results.csv)、[測定コードと手順](memory_probe_v10/README.md) を保存しています。
Flashは7.94〜31.26%削減、静的RAMは全条件で6,764 byteでした。紙の再測定は行っておらず、既存の10回測定結果は変えていません。

**紙の新規10回読取を受領・検算し、[実測に基づく報告書・コード・データ一式](../experiment_report_v10/README.md) を作成しました。**
全10回の [MSE平均・標本SD・元ログ](fresh_readings/README.md) を保存しています。高いMSEの第2回も含め、除外はありません。
10回平均はDense60・80・100が300以下、CSRは全Hで300超えです。以下の保存入力再生とは別の結果として扱います。

先生指定は元の **[result.ppm](reference/result.ppm)** です。CMYK版を勧めた以前の案内は、この課題の基準として誤りでした。
正解画像を変更せず、3×3の学習ラベルを先生指定RGBへ修正して学習を行いました。
元ファイルのバイト列を保存しています。SHA256は `582b783b099db5bd6fae89ee0355a17f427fb4a4fc3f2750a8acb8398e5383e2` です。

**2026年11月9日受領：全8条件の実機再生・4組の実機速度・比較プログラム全体のFlash/RAMを記録できました。**
[今回の実機結果表・図](RESULTS.md) と [元ログ・検算](hardware/run_01/README.md) にまとめています。
実機出力216成分はPCと完全一致しました。以下の操作手順は再現用です。

**サイトの左側は先生指定のresult.ppmのままにしてください。右側には新しいV10の出力を入れます。**
以前のresu15.ppmの画素は変わっていないため、そちらを再度アップロードすると以前の3471.00のままです。
過去のCMYK基準の値を先生指定基準の達成結果として扱うことは撤回します。

## まず先生のサイトで確認する

以下は、最新V9で実際に読み取った9色を保存し、**学習へ加えずに、修正後のNNでPC再計算した出力**です。
先生の元HTMLを変更せずに実行し、下表と一致することを確認しました。
新しく紙を読み取ったV10実測や、未見チャートの最終評価ではありません。
受領した [先生の計算HTML](reference/mse_calculator.html) も変更せず同梱しました。ブラウザでローカルに開いて確認できます。

| H | Denseの新出力とMSE | 枝刈り＋CSRの新出力とMSE |
| --- | ---: | ---: |
| 40 | [h40_dense.ppm](site_outputs/h40_dense.ppm) : 122.44 | [h40_csr.ppm](site_outputs/h40_csr.ppm) : 297.22 |
| 60 | [h60_dense.ppm](site_outputs/h60_dense.ppm) : 33.96 | [h60_csr.ppm](site_outputs/h60_csr.ppm) : 273.19 |
| 80 | [h80_dense.ppm](site_outputs/h80_dense.ppm) : 60.85 | [h80_csr.ppm](site_outputs/h80_csr.ppm) : 443.04 |
| 100 | [h100_dense.ppm](site_outputs/h100_dense.ppm) : 21.93 | [h100_csr.ppm](site_outputs/h100_csr.ppm) : 360.56 |

CSRのH80とH100は目標300を超えています。結果を見て同じ9色で閾値を選び直してはいません。
計算はRGB0〜255、9画素×3成分の二乗誤差の平均です。予測値を0〜1へクリップし、255倍して四捨五入します。
PPMはNNの回帰出力で、色番号による正解値の出力やパレット置換は行いません。

## ArduinoでMSE・容量・速度をまとめて記録する

コードと学習済みパラメータは次の2ファイルに分けています。

- [teacher_compare_v10.ino](teacher_compare_v10/teacher_compare_v10.ino)
- [teacher_parameters_v10.h](teacher_compare_v10/teacher_parameters_v10.h)

1. 両方を同じ **teacher_compare_v10** フォルダに保存して、Arduino IDEで `.ino` を開きます。
2. **Arduino UNO R4 WiFi** と接続ポートを選び、書き込みます。
3. シリアルモニタを **9600 bps** にしてRESETします。`FIRMWARE=TEACHER_RESULT_COMPARE_V10` と `ALL EIGHT MODELS CHECK: PASS` を確認します。
4. 半角の **tを1回** 送信します。黒白登録や紙の再読取は不要です。
5. `TEACHER RESULT COMPARISON START`〜`END` と `FOUR PAIR BENCHMARK START`〜`END` を保存します。IDEのFlash・グローバル変数使用量2行も保存します。

MSEはこの保存済み9色を実機へ入力した再生結果、速度は**今回のV10をUNO R4上で実際に計時した値**になります。
100推論×5組、ウォームアップ10回、Dense先行・CSR先行を交互にします。同じ9入力を両方式へ使い、センサ・シリアル・MSE計算は計時範囲から外します。
チェックサム加算は両方式の計時に含みます。8条件入りファームウェアのIDE使用量と、各条件の数値配列容量を区別します。
旧モデルの実機速度やV9のFlash158688byte・静的RAM7212byteを、V10の値として転記しません。

## 学習データと評価データ

- [training135.csv](training135.csv) は既存の実測135行を全行保持しています。入力値は変更していません。
- うち3×3の111行を、色IDに対応する先生指定result.ppmのRGBへ修正しました。`target_source` が現在の正解の出典です。
  `previous_target_*` と継承した `source_role` は過去の処理の記録で、そこに残るCMYKの記載は現在の正解を意味しません。
- 6×4の24行は元の配布Notebookの正解を保持しました。先生指定result.ppmは3×3なので、6×4まで先生のPPMで確認したとはしていません。
- [heldout9.csv](heldout9.csv) は最新V9実測の9入力です。親V9モデル・V10学習・損失重み・拡張幅・枝刈り閾値の選択には加えていません。
  ただし過去の失敗ログを見て設計した追試なので、同一学習チャートの保存入力による事後的な検証です。独立した最終評価とはしません。

[出典とハッシュ](provenance.json)、[正解の変更一覧](target_changes.csv)、[学習条件](training_protocol.json) を保存しました。
構造3→H→H→3、FP32、隠れ層ReLU、線形出力、同じHの固定V9から初期化しました。
各Hとも1条件だけ、seed42・40,000epoch、Adam0.0005→0.000005、正の色・校正グループ重み付きSmoothL1 beta0.05、重みだけのL1係数1e-6です。
V9と同じ学習入力のgain/offset拡張を用いました。保存した9色でepochや学習条件を選び直していません。

古い不安定な測定も残しているため、学習全体の通常の重みなしMSEは次の値です。
上の9色の結果を全135行や全33色で300以下だったと読み替えません。

| H | 学習135行のMSE | うち3×3の111行のMSE |
| --- | ---: | ---: |
| 40 | 1794.73 | 1049.10 |
| 60 | 1346.71 | 867.25 |
| 80 | 1151.08 | 764.50 |
| 100 | 922.77 | 495.82 |

## 枝刈り・CSRの選択

各Hの新しい重みの絶対値から全候補を作り、別に保存した検証128入力（8×8を2回）で、MSE増加5%以下・Dense未満の容量という従来条件を確認しました。
さらに、学習内のV8の9入力でMSE300以下という利用者目標を満たす候補から最小容量を選びました。
最新V9の保存9入力は選択へ使っていません。元の検証チャート2回目には異なる色が同一入力になる問題があるため、閾値は暫定です。
検証128入力の正解は従来の8×8チャートのままで、3×3のresult.ppmの正解で置換しません。

| H | 閾値 | NNZ | Dense配列byte | CSR配列byte |
| --- | ---: | ---: | ---: | ---: |
| 40 | 0.0608498063 | 470 | 7692 | 2854 |
| 60 | 0.0543417912 | 1141 | 16332 | 6449 |
| 80 | 0.0455471985 | 1649 | 28172 | 9229 |
| 100 | 0.0589320157 | 2374 | 43212 | 13094 |

配列byteにはFP32重み・バイアス・uint8列番号・uint16行ポインタを含めます。ファームウェア全体のFlashやRAMではありません。
[全候補と選択理由](selection)、[PCの8条件の結果](comparison_pc_summary.json) を保存しました。

## 検算と未実施の項目

[verify.py](verify.py) で、元のresult.ppmの完全一致、全111ラベル、入力135行の不変、全行の正の学習重み、ヘッダ・NPZ・Torch重みの一致を確認しました。
400入力×8条件でC++11とPC計算を比較し、RGB8は完全一致しました。
ASan/UBSanを有効にし、CSR索引、t/T各1回の8PPM、元の先生HTMLのMSE、4組の計時制御・交互順序・チェックサムも確認しました。
ホスト試験の時計は偽物で、実機速度として保存しません。[検算結果](verification.json) に区別しています。
同じ検算は、NumPy・PyTorch・g++・Node.jsのある環境で `python teacher_result_v10/verify.py` を実行すると再現できます。
この環境でのArduino CLI取得はHTTP403で失敗したため、上のverification.jsonは当時のホスト検算の記録です。
その後、利用者からUNO R4での実行ログとIDE使用量を受領しました。現在の実機結果は [hardware/run_01/summary.json](hardware/run_01/summary.json) です。
受領した実機PPM・元の先生HTML・計時の平均と標本SD・チェックサムを検算済みです。全8条件入りのFlash187888byte、静的RAM6740byteも記録しました。

物理的10回反復と各条件単独のFlash・静的RAM8条件は実施・集計済みです。独立した最終評価チャート、H50、搭載限界と実行時最大スタックは未実施です。
H80・100は利用者の追加要望です。詳しい進捗は [remaining_experiments.md](remaining_experiments.md) に記録しています。
