# 実際の10回測定に基づく報告書・コード・データ一式

[リポジトリの案内へ戻る](../README.md)

[報告書PDF](report.pdf)、[編集できるWord版](report.docx)、[Markdown本文](report.md)、[LaTeX本文](report.tex) を用意しました。
内容を理解するための資料は [本人用の解説PDF（8ページ）](explanation/guide.pdf) です。実験の目的、式、コード、MSE、容量、速度を順に説明しています。提出用はreport.pdfです。
目的・方法・MSE集計までを確認する [途中稿PDF](draft/report_draft.pdf) も用意しています．完成版から一部を省略した抜粋稿です．
学籍番号・氏名は元の計画書にある情報を引き継いでいます。
用語表と重複した解説を削り，実験条件・必要な式とコード・結果・考察を中心に整理しました．本文の句読点は「．」「，」へ統一し，測定値と結論は保持しています．
図表キャプションは中央揃え・句読点なしです．MSEとその標本SD・対応差は小数第2位まで表示し，第3位以降を切り捨てています．平均や増加率は切り捨て前の値から計算し，元CSVは保持しています．
使用関数の役割は本文で説明し，行ごとの説明は付録のコメント付きコードにまとめています．7.5の見出しは「今後の改善」です．
提出用PDFはLaTeX（XeLaTeX）から生成しています。[第6週資料と本文の対応](requirements/coverage.md) も別ファイルへ記録しました。講義資料は参考文献一覧へ入れていません。

実際の紙の測定10回を全て含めています。第2回は操作ミスに気づいていないとの回答を残し、削除していません。
容量・固定入力での推論速度は改善しました。CSRの10回平均MSE300以下・精度維持は達成していない結果として記載しています。
各条件単独のFlash・静的RAMは8条件を受領・集計済みです。独立した最終チャート、H50、搭載限界と最大スタック使用量は未実施です。
この資料は実施した内容をまとめたもので、当初計画の全項目完了や採点結果を保証するものではありません。

## 必要なファイル

- 紙を読んだコード：[teacher_fresh_v10.ino](code/teacher_fresh_v10/teacher_fresh_v10.ino)、[数値ヘッダ](code/teacher_fresh_v10/teacher_parameters_v10.h)
- 各次元の色画像：[MSE付きの画像一覧](color_images/README.md)。最後の10回目の8条件と、全10回の個別PNG／PPMを掲載。画像ごとのMSEと10回平均を分けています。
- 速度を計ったコード：[teacher_compare_v10.ino](code/teacher_compare_v10/teacher_compare_v10.ino)、[数値ヘッダ](code/teacher_compare_v10/teacher_parameters_v10.h)
- 各モデル単独のFlash・静的RAM：[8条件の結果](data/single_model_memory_results.csv)、[4組の削減率](data/single_model_memory_comparison.csv)、[受領文・検算](raw_logs/hardware/single_model_memory_run_01/README.md)、[測定手順・コード](code/memory_probe_v10/README.md)。記入用template.csvは実験結果とは別の空欄ひな形です。
- 全10回・8条件（MSE80個）：[MSE一覧](data/physical10_mse_all80.csv)、[平均・標本SD](data/physical10_mse_summary.csv)、[同じ回の圧縮前後差](data/physical10_paired_differences.csv)
- 学習データ：[training135.csv](data/training135.csv)、[行重み](data/training_sample_weights.csv)
- 閾値選択の別入力：[threshold_validation128.csv](data/threshold_validation128.csv)
- 速度・容量：[集計CSV](data/fixed_input_device_timing_capacity.csv)
- 正解：[元のresult.ppm](reference/result.ppm)、[元の先生HTML](reference/mse_calculator.html)
- 各回の元ログ・80枚のPPM/PNG：[raw_logs/physical10](raw_logs/physical10)
- 速度の5組とIDE使用量の元ログ：[raw_logs/hardware](raw_logs/hardware)
- 図：[比較の流れ](figures/comparison_flow.png)、[10回のMSE](figures/physical_mse10.png)、[容量・速度](figures/capacity_and_time.png)、[単独モデルのFlash](figures/single_model_flash.png)。同名PDFも同梱。

コードと学習済み数値は別ファイルです。Arduinoでは使う版の2ファイルを同名フォルダに置きます。
元のデータは変更せず複製し、manifest.jsonへ各ファイルのSHA256を保存しています。
このフォルダで `python verify_package.py` を実行すると、全80MSE・10回の平均と標本SD・コードの対応・ファイルハッシュを再確認できます。

## 資料指定の提出日

2026年11月13日：作成中の実験報告書（書けている部分まで）、12月4日：完成原稿、12月11日：最終版。
資料ではTAへ提出して添削を受け、その指摘を反映する工程も含みます。提出後の指摘はこの原稿へ反映します。
提出用はreport.pdf、LaTeXソースはreport.texです。Word版は編集補助として同梱しています。

## 保存ログの再計算

Python3.12、NumPy・PyTorch・Pillow・Matplotlib、Node.js、C++11対応g++を使います。
requirements.txtはクラウドで使った主要Python版です。OS・CPUが違う場合は、凍結済みヘッダと受領した実機PPMを結果の照合基準にします。

source_snapshotを作業ディレクトリとして、次を実行できます。

```sh
python teacher_result_v10/record_fresh_capture.py teacher_result_v10/fresh_readings/capture_10/submitted_log.txt --received-date 2026-11-09
```

既に各回の検算済みsummaryと出力も同梱しています。上のコマンドは最後の回を再検算し、全10回を集計し直します。
数値ヘッダのモデル検査は `python teacher_result_v10/verify.py`、新規読取プログラムのホスト検査は `python teacher_result_v10/verify_fresh.py` です。
後者の模擬入力は実験の測定回数に含めません。

## 学習・枝刈りの使用ソース

source_snapshot内のteacher_result_v10にprepare.py、train.py、data_weights.py、select_compression.py、finalize_compression.py、build_program.py、build_fresh.pyを同梱しています。
学習を開始した親V9のチェックポイントと、最終V10のチェックポイント・NPZ・Denseヘッダ・CSRヘッダ・全候補CSVも含みます。
凍結済み成果物の誤上書きを避けるガードがあるため、そのままtrain.pyや選択処理を再実行する用途には使わず、再学習時は別の作業コピーを用意します。
記録済み結果の検算には再学習は不要です。
