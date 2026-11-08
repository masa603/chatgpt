# V9のDense／CSRを1回で比較する

実行コードは [compare_all_v9.ino](compare_all_v9.ino)、学習済み数値配列は [comparison_parameters_v9.h](comparison_parameters_v9.h) です。
この2ファイルを同じ `compare_all_v9` フォルダに保存し、.inoを開いてUNO R4 WiFiへ書き込みます。

1. シリアル9600でRESETし、`FIRMWARE=COLOR_COMPARE_V9` と `ALL EIGHT MODELS CHECK: PASS` を確認。
2. 黒b→白wで校正。
3. 赤・緑・青／黄・水色・紫／黒・灰・白を各rで1回ずつ読む。
4. `PAIRED CAPTURE START`〜`PAIRED CAPTURE END` を保存。
5. tを1回送信し、`FOUR PAIR BENCHMARK START`〜`FOUR PAIR BENCHMARK END` を保存。
6. Arduino IDEのFlash・グローバル変数使用量の2行も、全8条件入りプログラムの値として保存。

1回の9色読取で、4H×Dense/CSRの全8条件を同じ入力で計算します。tでも4組まとめて計測します。
新しい実測のMSE300以下は未確認です。計画書の主目的は圧縮前後の比較で、300という数値は利用者が追加した目標です。
追加測定はnで黒白の校正から開始できます。計画書の10回反復が完了したという扱いにはしません。
CSRは検証入力の衝突があるため暫定、独立した最終評価は未実施です。
[学習・閾値選択・検算の記録](../README.md) を保存しています。
