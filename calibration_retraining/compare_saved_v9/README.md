# 今回の9色を使い、tだけでDense／CSRと速度を比較する

[compare_saved_v9.ino](compare_saved_v9.ino) と [saved_parameters_v9.h](saved_parameters_v9.h) を、
同じ `compare_saved_v9` フォルダへ保存し、.inoをArduino IDEで開いてUNO R4 WiFiへ書き込みます。

1. シリアルモニタ9600 bpsでRESET。
2. **FIRMWARE=COLOR_SAVED_COMPARE_V9** と **ALL EIGHT MODELS CHECK: PASS** を確認。
3. 半角の **t** を1回送信。
4. **SAVED CAPTURE COMPARISON START〜END** と **FOUR PAIR BENCHMARK START〜END** を保存。
5. このスケッチを書き込んだ時のIDEの「スケッチ…バイト」「グローバル変数…バイト」の2行も保存。

黒白の登録や紙の読み直しは行いません。入力は利用者が今回読み取ったV9 CAPTURE=1の9色です。
Denseの108成分が元の実測出力と完全一致することを、CSRの結果を出す前に確認します。
重み・閾値は変更していません。新しい物理測定として数えず、同じ入力を実機で再計算した比較として記録します。
速度は既定の検証128入力を使い、4組とも10回ウォームアップ、100推論×5組、先行方式を交互にして測ります。

旧 **COLOR_RETRAIN_V9** はDenseで紙を読むための版で、tの処理はありません。
今使うのは、この **COLOR_SAVED_COMPARE_V9** です。
出力のMSEは配布CMYK版PPMが基準です。CSRも300以下と保証するものではなく、超えた条件や精度悪化も残します。

[保存した元の実測](../physical_readings/capture_01/README.md)、[ホストでの検算](../saved_comparison_verification.json) があります。

## サイトで3471になる場合

サイト左の純色RGBのresult.ppmと、モデルが学習した配布NotebookのCMYK画像では正解RGBが違います。
今回のH40出力は前者で3471.00、後者で278.33です。どちらも先生のHTMLで再現しました。
このt版では同じ予測に対する両基準のMSEを並べて表示します。
300以下の判定欄の基準は選択済みの配布CMYK版です。純色result.ppm基準でも達成したとは扱いません。
配布Notebookの基準をサイトへ入れる場合は [CMYK版PPM](../../distributed_training/references/reference_image1_cmyk_3x3.ppm) を使います。
