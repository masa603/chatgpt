# 40・60・80・100次元のV8：1ファイル版

[color_retrain_v8_single.ino](color_retrain_v8_single.ino) に学習済み配列も全て含めています。
ファイルを開いて全コードを新規スケッチへ貼り付けるか、同名フォルダへ保存してArduino IDEで開きます。

UNO R4 WiFiへ書き込み、シリアル9600 bpsでRESETします。
`FIRMWARE=COLOR_RETRAIN_V8` と `ALL MODELS CHECK: PASS` を確認してください。
黒b→白wで校正し、赤・緑・青／黄・水色・紫／黒・灰・白を各rで1回ずつ読みます。
**PATCH ADAPTED CAPTURE START〜PATCH ADAPTED CAPTURE END**を保存します。

1回で4モデルの結果が出ます。今回の15件を学習へ加えましたが、新しいV8実測MSEは未測定です。
先生のサイトの基準は配布CMYK版PPMです。[学習・開発・検算の全記録](../README.md) があります。
