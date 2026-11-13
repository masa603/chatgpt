# 色補正実験：最新版の案内

**提出する報告書は [report.pdf](experiment_report_v10/report.pdf) です。**
2026年11月13日更新。第6週資料の指定内容を反映した報告書に加え、本人が式・コード・結果を理解するための解説PDFを用意しました。

| やりたいこと | 開く場所 |
| --- | --- |
| 提出用の報告書を見る | [報告書PDF](experiment_report_v10/report.pdf) |
| レポートの内容を理解する | [本人用の解説PDF（8ページ）](experiment_report_v10/explanation/guide.pdf) |
| LaTeXで編集する | [report.tex](experiment_report_v10/report.tex) |
| Wordで編集する | [report.docx](experiment_report_v10/report.docx) |
| 各次元の色画像とMSEを見る | [色画像の一覧](experiment_report_v10/color_images/README.md) |
| 使ったArduinoプログラムを見る | [code](experiment_report_v10/code) |
| 学習データを見る | [training135.csv](experiment_report_v10/data/training135.csv) |
| 10回測定の平均MSEを見る | [MSE集計](experiment_report_v10/data/physical10_mse_summary.csv) |
| 実験データをまとめて見る | [data](experiment_report_v10/data) |
| 元の測定ログを確認する | [raw_logs](experiment_report_v10/raw_logs) |

## 今見るフォルダ

```text
README.md                 ← この案内
experiment_report_v10/    ← 最新版の報告書・コード・データ一式
archive/                  ← 開発履歴・再計算用ソース・別の課題
```

最新版の内容と再計算方法は [報告書一式の説明](experiment_report_v10/README.md) にまとめています。
過去のV3〜V9や旧MSE、以前の計画書を調べるときは [archiveの案内](archive/README.md) を開いてください。

## Arduinoのコードを選ぶ

| 用途 | フォルダ |
| --- | --- |
| 紙を読み、40・60・80・100次元のDense／CSRを比較した版 | [teacher_fresh_v10](experiment_report_v10/code/teacher_fresh_v10) |
| 保存入力でMSEと推論速度を測った版 | [teacher_compare_v10](experiment_report_v10/code/teacher_compare_v10) |
| 1モデルずつFlash・静的RAMを測った版 | [memory_probe_v10](experiment_report_v10/code/memory_probe_v10) |

各フォルダの `.ino` と `.h` を同じフォルダに置いて開きます。学習データは [data](experiment_report_v10/data) に分けてあります。
測定済みなのは、紙の読取10回と単独メモリ8条件です。これらの元データは保持しています。
