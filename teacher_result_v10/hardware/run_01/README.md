# V10実機再生・速度・IDE使用量（2026年11月9日受領）

[結果表・図・考察](../../RESULTS.md) に集計しています。
保存済みの実測9入力をArduino UNO R4 WiFi上で8モデルへ入力した結果と、実機の速度測定です。
新しいセンサ読取は0回です。元の学習チャートの読取データを再利用した比較で、独立した最終評価ではありません。

- submitted_log.txt：利用者が送ったシリアル表示。チャット上のバックスラッシュを保持。
- normalized_log.txt：行頭・行末の表示用バックスラッシュを除いた解析用ログ。
- compile_submitted.txt：今回のIDE使用量・書込表示。
- summary.json：全検算・実機集計・データの出典。
- memory.json：Flash187888byte、静的RAM6740byte、残るRAM26028byte。8条件入り全体の値。
- mse.csv、rgb9.csv、timing_groups.csv：8条件のMSE、画素、計時5組の生データ。
- h40/60/80/100_dense/csr.ppm・png：受領した実機出力の保存画像。

216成分のRGB8がPC出力と完全一致し、先生の元HTMLのMSEも一致しました。
4組の速度の平均・標本SD・短縮率と交互の計時順序、チェックサムも検算済みです。
CSRのH80・100はMSE300超え、全HでDenseよりMSEが増えた結果をそのまま残しています。
