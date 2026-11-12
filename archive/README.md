# 開発履歴・再計算用ソースの保存先

[最新版の案内へ戻る](../README.md)

2026年11月12日にリポジトリのトップを整理し、作業フォルダと別の課題をここへ移動しました。
**現在使う報告書・Arduinoコード・集計データは [experiment_report_v10](../experiment_report_v10/README.md) にあります。**

## 保存しているもの

| 内容 | 保存先 |
| --- | --- |
| 最終V10の学習・枝刈り・検算・報告書生成用ソース | [teacher_result_v10](teacher_result_v10/README.md) |
| 最終モデルの親V9と校正への対応 | [calibration_retraining](calibration_retraining/README.md) |
| V7・V8の開発と追加学習 | [distributed_training](distributed_training/README.md)、[patch_retraining](patch_retraining/README.md) |
| V3〜V6の開発・正解画像の照合 | [learning_followup](learning_followup/README.md)、[reference_corrected](reference_corrected/README.md)、[current_input_adaptation](current_input_adaptation/README.md)、[data_learning_followup](data_learning_followup/README.md) |
| 初期の学習・枝刈り・実機比較 | [retraining](retraining/README.md)、[pruning](pruning/README.md)、[experiment_today](experiment_today/README.md) |
| 元のスキャナと初期の実装 | [hdjsk](hdjsk/README.md)、[hdjsk_csr](hdjsk_csr/README.md)、[hdjsk_benchmark](hdjsk_benchmark/README.md)、[hdjsk_retrained](hdjsk_retrained/README.md) |
| ボタン・センサ・校正の確認 | [diagnostics](diagnostics)、[calibration_followup](calibration_followup/README.md) |
| 以前の計画書 | [PDF](experiment_plan.pdf)、[LaTeX](experiment_plan.tex) |
| 別の授業課題 | [CE5](CE5/README.md)、[assignment4.py](assignment4.py)、[assignment4のPDF](assignment4_solution.pdf)、[LaTeX](assignment4_solution.tex) |

古い版のREADMEには、その時点の手順・正解画像・MSEが残っています。現在の結果は最新版の報告書で確認できます。
旧MSEや正解画像の履歴を、最新版の評価結果に混ぜないようにしています。

## 再計算する場合

元の作業フォルダ同士の位置関係を保つため、同じ階層のまま移動しています。
以前のコマンドを実行するときは、**このarchiveフォルダを作業ディレクトリ**にします。

```sh
cd archive
python teacher_result_v10/verify.py
```

報告書の生成は `python teacher_result_v10/build_report_package.py` です。必要なPythonライブラリ・pandoc・XeLaTeX・フォントは最新版READMEの説明を参照してください。
出力先はリポジトリ直下の `experiment_report_v10` です。
既存の重み・閾値・測定ログを確認するだけなら、再学習は不要です。

[移動前後のパス一覧](moved_paths.json) と、[整理前後のファイル・リンク検査結果](organization_check.json) を保存しています。
