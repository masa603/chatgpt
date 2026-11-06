# MSEの比較条件を確認中

利用者から「先生の計算サイトではMSEが約3000になる」と指摘があり、
先生の基準で300以下とした判定を取り下げました。
**93.85などの値は、プログラム内の復元済み9色に対する参考値です。**
利用者がサイトに選んだ基準PPM・比較PPMを確認するまで、正式な目標達成は未確認です。
学習済み重み、学習ラベル、元ログ、出力画像は変更していません。

## 提供HTMLから確認できた内容

提供された `mse_calculator.html` のSHA256は [calculator_check.json](calculator_check.json) に保存しています。
RGB各成分の二乗誤差の平均で、計算式は `sum((R-Rref)^2+(G-Gref)^2+(B-Bref)^2)/(画素数*3)` です。
同じ寸法・同じRGBのPPMを比較すれば、従来こちらが用いた式と一致します。
P3形式を読み、RGBをCanvasのUint8ClampedArrayへ直接入れます。最大値の正規化は行いません。

提供HTMLのJavaScriptを変更せず実行し、こちらが復元した9色の3×3 PPMと実測PPMの組では
H40=524.07、H60=93.85、H100=119.93になることを確認しました。
これは先生のサイトに利用者が実際に入れた基準PPMとの照合ではありません。
こちらの復元9色はNotebookの `reference_image1_cmyk_large.png` の保存済み表示から採取した値です。
元PNGや、先生のサイトで選ばれたPPMそのもののRGBとの一致は未確認です。

また、提供HTMLは寸法が違う画像でもエラーにせず、配列先頭の
`min(width1,width2)*min(height1,height2)` 画素を比較します。
画像の拡大縮小や行幅を合わせる処理はありません。寸法不一致の場合は画像全体の対応比較にはなりません。
今回その問題が起きたかどうかも、実際にサイトへ入れた2ファイルがないため未確定です。

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
