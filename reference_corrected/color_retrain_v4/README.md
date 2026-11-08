# 正解PPM修正版 V4

このフォルダの `color_retrain_v4.ino` と `trained_models_v4.h` を同じフォルダに保存し、
UNO R4 WiFiへ書き込んでください。

9600 bpsでRESETすると **FIRMWARE=COLOR_RETRAIN_V4** と表示されます。
**ALL MODELS CHECK: PASS** の後、黒でb、白でw、9色を左上から各rで読み取ります。
順番は赤・緑・青／黄・水色・紫／黒・灰・白です。

全9色の後の **CORRECTED CAPTURE START〜CORRECTED CAPTURE END** を保存します。
この1回で40・60・100次元を比較します。先生のサイトの基準PPMは元の `result.ppm` を使います。
新しい実測で300以下かどうかは未確認です。
学習条件と比較基準を直した理由は [上位の記録](../README.md) にあります。
