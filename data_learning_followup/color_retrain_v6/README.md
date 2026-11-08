# 学習データ側を調整したV6

`color_retrain_v6.ino` と `trained_models_v6.h` を同じフォルダに置き、UNO R4 WiFiへ書き込みます。
9600 bpsでRESETし、**FIRMWARE=COLOR_RETRAIN_V6** と **ALL MODELS CHECK: PASS** を確認します。

黒でb、白でwの後、赤・緑・青／黄・水色・紫／黒・灰・白を各rで読み取ります。
全9色後の **LEARNING CAPTURE START〜LEARNING CAPTURE END** を保存してください。
1回で40・60・100次元のMSEとPPMが出ます。正解画像は元のresult.ppmのままです。

最新の9組も最終学習に含めたため、精度はさらに新しい読取で確認します。
学習誤差は小さくなりましたが、開発検証はまだ300を超え、新しいV6実測は未測定です。
[学習・検証の全記録](../README.md) を保存しています。
