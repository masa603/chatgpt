# MSEの比較基準の取り違えを確認

利用者から「先生の計算サイトではMSEが約3000になる」と指摘があり、
先生の基準で300以下とした判定を取り下げました。
その後 [result.ppm](provided/result.ppm) と [resu12.ppm](provided/resu12.ppm) を受領し、
**先生のサイトのJavaScriptでMSE3984.00を再現しました。** 両方3×3で、サイズの問題ではありませんでした。
旧版が学習・比較に使ったCMYK画像由来のRGBと、提供基準PPMの純色RGBが異なっていたことが原因です。
旧ファイルは保持し、[reference_corrected](../../reference_corrected/README.md) に正解PPMで再学習したV4を作成しました。

| V3のH | 旧内部基準とのMSE | 提供result.ppmとのMSE | 300以下 |
| --- | ---: | ---: | --- |
| 40 | 524.07 | 3984.00 | 未達 |
| 60 | 93.85 | 3442.59 | 未達 |
| 100 | 119.93 | 3522.89 | 未達 |

`resu12.ppm` のRGB27成分はV3のH40出力と一致します。
H60やH100で比較しても上表のとおり300を超えており、以前の達成報告を訂正します。
元ファイルをバイト単位で保存し、計算とSHA256は [provided_pair_check.json](provided_pair_check.json)、
基準色の相違は [target_rgb_difference.csv](target_rgb_difference.csv) に記録しています。

## 提供HTMLから確認できた内容

提供された `mse_calculator.html` のSHA256は [calculator_check.json](calculator_check.json) に保存しています。
RGB各成分の二乗誤差の平均で、計算式は `sum((R-Rref)^2+(G-Gref)^2+(B-Bref)^2)/(画素数*3)` です。
同じ寸法・同じRGBのPPMを比較すれば、従来こちらが用いた式と一致します。
P3形式を読み、RGBをCanvasのUint8ClampedArrayへ直接入れます。最大値の正規化は行いません。

提供HTMLのJavaScriptを変更せず実行し、こちらが復元した9色の3×3 PPMと実測PPMの組では
H40=524.07、H60=93.85、H100=119.93になることを確認しました。
これは旧プログラムの内部基準との算術確認で、提供された正解PPMとの比較とは分けています。
こちらの復元9色はNotebookの `reference_image1_cmyk_large.png` の保存済み表示から採取した値です。
その後に提供された基準PPMとはRGBが異なることを確認しました。
例えば旧教師データの緑は105,189,69、提供基準の緑は0,255,0でした。

また、提供HTMLは寸法が違う画像でもエラーにせず、配列先頭の
`min(width1,width2)*min(height1,height2)` 画素を比較します。
画像の拡大縮小や行幅を合わせる処理はありません。寸法不一致の場合は画像全体の対応比較にはなりません。
今回受領した2ファイルは両方3×3なので、この寸法不一致の問題は該当しません。

## 再現用スクリプト

[run_teacher_calculator.cjs](run_teacher_calculator.cjs) は提供HTMLのscriptをそのまま実行します。
入力のサイズが違う場合、サイトの表示値を再現するとともに比較条件の警告も記録します。
提供HTMLそのものはこのリポジトリへ再配布していません。

```sh
node learning_followup/reference_audit/run_teacher_calculator.cjs mse_calculator.html 基準.ppm 比較.ppm
```

黒同士のMSE=0、黒と白のMSE=65025、RGB3成分での平均、寸法不一致を検査しました。
[embedded_reference_v3.ppm](embedded_reference_v3.ppm) は既存プログラム内の比較値を明示するために出力したものです。
**先生の基準画像の代わりとして使い、低いMSEを正式結果にするためのファイルではありません。**
