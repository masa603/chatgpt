# 配布画像を正解にした40・60・80・100次元（V7）

`color_retrain_v7.ino` と `trained_models_v7.h` をこのフォルダに置いて、UNO R4 WiFiへ書き込みます。
`reference_image1_cmyk_3x3.ppm` は先生の計算サイトの基準画像です。
古い純色RGBのresult.ppmとはRGBが違うため、基準を今回のCMYK版へ切り替えてください。

9600 bpsでRESETし、**FIRMWARE=COLOR_RETRAIN_V7** と **ALL MODELS CHECK: PASS** を確認します。
黒b→白wで校正し、赤・緑・青／黄・水色・紫／黒・灰・白の各色でrを1回ずつ送ります。
**DISTRIBUTED CAPTURE START〜DISTRIBUTED CAPTURE END** を丸ごと保存してください。
1回で4モデルを同じ新規入力で計算します。80次元を含む新しい実測のMSE300以下は未確認です。

開発用9色のPC比較は300以下でしたが、条件選択に使ったデータです。
実測での達成は新しい読み取りで確認します。[学習・検証の全記録](../README.md) を保存しています。
