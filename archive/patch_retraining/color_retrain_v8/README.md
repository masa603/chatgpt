# V8の2ファイル版

**現在は [V9の全8条件比較版](../../calibration_retraining/compare_all_v9/README.md) を使います。以下は旧V8の操作記録です。**

利用者の希望に従い、次のように分けています。

- [color_retrain_v8.ino](color_retrain_v8.ino)：センサ読取・NN推論・MSE計算などの実行コード。
- [trained_models_v8.h](trained_models_v8.h)：学習済みの重み・バイアス、正解RGBなどの数値配列。
- [training126.csv](../training126.csv)：PCでの学習に使った元データ。

`.ino` と `.h` を同じ `color_retrain_v8` フォルダに保存し、Arduino IDEで `.ino` を開きます。
CSVをスケッチへ貼り付ける必要はありません。

UNO R4 WiFiへ書き込み、シリアル9600 bpsでRESETします。
黒b→白wで校正し、赤・緑・青／黄・水色・紫／黒・灰・白を各rで読みます。
**PATCH ADAPTED CAPTURE START〜PATCH ADAPTED CAPTURE END**を保存してください。

[1ファイル版](../color_retrain_v8_single/README.md) と計算・シリアル出力は同じです。
両方の形式を同じスケッチへ混ぜず、どちらかの形式だけを使います。
操作と結果の区分は [学習・検算の記録](../README.md) にあります。
