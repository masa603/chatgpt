# 新候補の確認時に届いた旧版の読取ログ

`old_firmware_retry_01/raw_log.txt` は2026年11月6日に追加で受領した9色の読取です。
新候補の測定を案内した後の受領ですが、ログに新候補の識別表示がなく、
閾値が旧候補の0.1387345567／0.1484270990／0.2595954686のため、旧版の結果と識別しました。
正確な取得日時は未確認で、受領順と実行時刻を混同しません。

原文、6枚のPPM、校正値・生値・NN入力・参照RGB、構造化記録を保存しました。
6枚のPPMから計算したMSE、正規化、旧モデルの出力が一致しています。
ソフトウェアの版が違うため新候補の実機測定には数えませんが、数値が悪いことを理由に削除しません。

当初2回として確定した旧版の集計は保存し、追加データは区別します。
[all_received_old_model_mse.csv](all_received_old_model_mse.csv) に旧版の有効な全3ログ、
[all_received_old_model_summary.csv](all_received_old_model_summary.csv) に追加分を含む参考集計を残します。
CAPTURE番号は装置の再起動で1に戻るため、ログのパスで各測定を識別します。
CAPTURE=2の色の読取ミスとして除外した測定は、今回の追加ログとは別です。

生値はモデルの重みとは独立して取得するため、今回の9入力を新候補へ共通で与える比較に利用できます。
候補はこのログの受領前に固定済みで、今回の入力で重みや閾値を選び直していません。
新しい [precision_scan](../precision_followup/precision_scan/README.md) の `v` でArduino上の再計算を確認します。
これは保存入力の実機再生であり、v送信時に紙を新しく読む測定ではありません。
同じ学習チャートの再読取なので、独立した最終評価には当たりません。

原文からの検算は次で行います。既定の2回集計へ自動追加しないよう明示しています。

```sh
python3 experiment_today/record_fresh_capture.py experiment_today/version_mismatch/old_firmware_retry_01/raw_log.txt --received-date 2026-11-06 --no-aggregate
```
