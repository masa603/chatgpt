# 先生指定result.ppmで比較するV10

**全8条件の実機再生と4組の速度、全体Flash/RAMは受領・検算済みです。[実機結果表・図](../RESULTS.md) に保存しました。**
以下は同じ比較を再現する操作です。

[teacher_compare_v10.ino](teacher_compare_v10.ino) と [teacher_parameters_v10.h](teacher_parameters_v10.h) を同じteacher_compare_v10フォルダに置きます。

Arduino UNO R4 WiFiへ書き込み、シリアル9600 bpsでRESETし、`FIRMWARE=TEACHER_RESULT_COMPARE_V10` と `ALL EIGHT MODELS CHECK: PASS` を確認します。
**tを1回送信**すると、保存済み実測9入力でDense／枝刈り＋CSRの8条件のMSE・PPMと、4組の速度を表示します。

先生指定のresult.ppmを変えず、そのRGBを正解として再学習した新しい重みです。
紙の再読取・b/w校正はありません。保存入力の実機再生と新規の物理読取を区別します。
`TEACHER RESULT COMPARISON` と `FOUR PAIR BENCHMARK` の両ブロック、IDE使用量2行を保存してください。

PCでの保存入力のMSEはDense40/60/80/100が122.44/33.96/60.85/21.93、CSRは297.22/273.19/443.04/360.56です。
CSR80/100は300超えです。受領した実機のRGB8はPCと完全一致しました。新たなセンサ読取ではなく、保存入力の実機再生です。
元の先生サイトの左側はresult.ppmのままで、右側には [新しい出力](../site_outputs) を入れます。
古いresu15.ppmを再使用しても、その出力画素は変わりません。

[学習データ・出典・全結果](../README.md)
