# 低い入力域を追加したV5

`color_retrain_v5.ino` と `trained_models_v5.h` を同じフォルダに置き、UNO R4 WiFiへ書き込みます。
9600 bpsでRESETし、**FIRMWARE=COLOR_RETRAIN_V5** と **ALL MODELS CHECK: PASS** を確認します。

今の照明条件を保ち、黒b→白wで校正し、赤・緑・青／黄・水色・紫／黒・灰・白の順に各rで読みます。
全9色の後の **ADAPTED CAPTURE START〜ADAPTED CAPTURE END** を保存してください。
1回で40・60・100次元のMSEとPPMが出ます。正解は元のresult.ppmのままです。

今回までのログを学習に使ったため、新しい読取の結果で精度を判断します。
読取回を外した開発検証でも誤差が大きく、新しい実測1回でも3モデルとも300を超えました。
次は装置の配置と校正値の変動を確認する段階です。新しいプログラムへの書き換えは案内していません。
学習条件・全検証結果は [上位の記録](../README.md) にあります。
