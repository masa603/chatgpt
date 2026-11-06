# 新候補の識別と保存入力の実機再生

旧版と区別するため、フォルダ・スケッチ・ヘッダの名前を変えました。
この2つを同じ **precision_scan** フォルダに保存します。

- [precision_scan.ino](precision_scan.ino)
- [precision_models.h](precision_models.h)

1. Arduino IDEで `precision_scan.ino` を開き、UNO R4 WiFiへ **→（書き込み）** します。
2. シリアルモニタを **9600 bps** にしてRESETを押します。
3. 最初に **`FIRMWARE=PRECISION_SCAN_V2`** が表示され、続いて **`ALL MODELS CHECK: PASS`** になることを確認します。
4. 半角の **`v`** を送信します。
5. **`PRECISION REPLAY START`〜`PRECISION REPLAY END`** を保存します。

新版のCSR配列容量は40次元4,774 byte、60次元4,424 byte、100次元24,304 byteです。
閾値は順に0.0011764123、0.0020613672、0.0011608397です。
旧版の閾値0.1387／0.1484／0.2596は、このスケッチでは使用しません。

`v` は今回受領した旧版の追加読取の生値・校正値から復元した9入力を、新しい6モデルへ与えます。
6条件のMSEとPC参照値との数値一致を表示し、センサは読み取りません。
データの出典は [latest_replay_manifest.json](../latest_replay_manifest.json) と
[旧版追加ログ](../../version_mismatch/README.md) に記録しています。
閾値の選択には使っていない入力ですが、学習済みの9色であり独立した最終評価ではありません。
物理的な取得とArduino上の再計算を別の処理として記録します。

この確認後に別の新しい紙の読取を行う場合は、同じスケッチで黒 `b`・白 `w` を登録し、
左上から9色を `r` で読みます。CAPTURE番号は1から始まり、後続の読取は `n` で開始できます。
元の2回分とは新候補の段階として区別して保存します。
