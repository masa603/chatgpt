# 学習データ側を調整したV6

**現在は [V9のDense／CSR・全8条件比較版](../../calibration_retraining/compare_all_v9/README.md) を使います。**
以下は旧V6の操作記録です。

`color_retrain_v6.ino` と `trained_models_v6.h` を同じフォルダに置き、UNO R4 WiFiへ書き込みます。
9600 bpsでRESETし、**FIRMWARE=COLOR_RETRAIN_V6** と **ALL MODELS CHECK: PASS** を確認します。

黒でb、白でwの後、赤・緑・青／黄・水色・紫／黒・灰・白を各rで読み取ります。
全9色後の **LEARNING CAPTURE START〜LEARNING CAPTURE END** を保存してください。
1回で40・60・100次元のMSEとPPMが出ます。正解画像は元のresult.ppmのままです。

最新の9組も最終学習に含めたため、精度はさらに新しい読取で確認します。
学習誤差は小さくなりましたが、受領したV6実測はH40=2483.26、H60=2233.85、H100=2884.96で、全3モデルとも300以下は未達でした。
[学習・検証の全記録](../README.md) を保存しています。
