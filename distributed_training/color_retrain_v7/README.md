# 配布画像を正解にした40・60・80・100次元（V7）

**現在は [追加15件を反映したV8の1ファイル版](../../patch_retraining/color_retrain_v8_single/README.md) を使います。**

`color_retrain_v7.ino` と `trained_models_v7.h` をこのフォルダに置いて、UNO R4 WiFiへ書き込みます。
`reference_image1_cmyk_3x3.ppm` は先生の計算サイトの基準画像です。
古い純色RGBのresult.ppmとはRGBが違うため、基準を今回のCMYK版へ切り替えてください。

9600 bpsでRESETし、**FIRMWARE=COLOR_RETRAIN_V7** と **ALL MODELS CHECK: PASS** を確認します。
黒b→白wで校正し、赤・緑・青／黄・水色・紫／黒・灰・白の各色でrを1回ずつ送ります。
**DISTRIBUTED CAPTURE START〜DISTRIBUTED CAPTURE END** を丸ごと保存してください。
1回で4モデルを同じ新規入力で計算します。

受領したV7実測はH40=3535.48、H60=3060.11、H80=3428.74、H100=3636.74で、全4モデルとも300以下は未達でした。
開発用9色のPC比較は300以下でしたが、その値は今回の新しい読取には再現しませんでした。
追加の15件は受領済みで、[V8](../../patch_retraining/README.md) の学習へ使用しました。
[学習・検証の全記録](../README.md) を保存しています。
