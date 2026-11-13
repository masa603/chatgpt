---
title: "枝刈り・CSRによる色補正モデルの圧縮"
subtitle: "アジャイルワーク2　実験報告書"
author: "@@ID@@　@@NAME@@"
student-id: "@@ID@@"
student-name: "@@NAME@@"
date: "記録整理日：2026年11月13日"
---

# 概要

色補正ニューラルネットワークに枝刈りとCSRを適用し，Denseと容量・推論時間・MSEを比較した．隠れ層の大きさHを40・60・80・100とし，9色のチャートを10回読み取った．各回の同じ入力を全8条件へ与えた．

モデル配列容量は60.51〜69.70%，固定入力による実機推論時間は28.97〜44.01%，単独モデル入りプログラムのFlashは7.94〜31.26%減少した．静的RAMは全8条件で6,764 byteだった．一方，10回平均MSEはDenseが@@DENSE_MEAN_RANGE@@，枝刈り＋CSRが@@CSR_MEAN_RANGE@@となり，容量・速度の改善と精度維持の両立は達成しなかった．

# 目的

Arduino上の色補正モデルについて，枝刈り＋CSRによる保存容量と推論時間の削減効果，および補正精度への影響を調べる．Denseに対するMSE増加5%以下を精度維持の条件とし，MSE300以下も別に判定する．

同じHのDenseと枝刈り＋CSRへ同一入力を与えて比較した．当初計画の40・50・60次元から，実施条件を40・60・80・100へ変更した．H50は未実施である．

# 比較の構成

図1のように，1回のセンサ読取から得たRGBを両方式へ入力した．4種類のHと2方式を組み合わせ，比較条件は8個とした．

\Needspace{14\baselineskip}

```{=latex}
\noindent\begin{minipage}{\linewidth}
```

![比較の流れ](figures/comparison_flow.png)

図1　同一入力による比較

```{=latex}
\end{minipage}
```

| 用途 | データ・操作 |
| --- | --- |
| 学習 | 過去の実測135行：3×3チャート111行，6×4チャート24行 |
| 閾値選択 | 別の8×8チャートを2回読んだ128入力と，学習内9入力 |
| 精度測定 | 学習にも使用した3×3チャートを新しく10回読取，計90入力 |
| 速度測定 | 保存済み実測9入力を実機で反復再生 |
| メモリ比較 | 各モデル単独の8条件をコンパイル |

新規10回の入力は学習・閾値選択へ加えていない．ただし，色自体は学習時と同じであり，独立した未見チャートによる最終評価ではない．

# 仕組みと計算方法

## ネットワークと学習

センサの読取値を黒白校正で正規化し，3−H−H−3の全結合ネットワークへ入力した．図2は層の構成と，計算単位1個の処理を示す模式図である．Hは各隠れ層の計算単位数であり，入力・出力はRGBの3成分である．

\Needspace{18\baselineskip}

```{=latex}
\noindent\begin{minipage}{\linewidth}
```

![ネットワークと計算単位](figures/theory_network.png){width=95%}

図2　ネットワークと計算単位

```{=latex}
\end{minipage}
```

RGB成分cの読取値を $v_c$，黒・白の校正値を $v_{B,c},v_{W,c}$ とすると，入力は式[正規化](#eq:normalization)で求める．clipは範囲外の値を0または1へ収める処理である．

::: {#eq:normalization .numbered-equation}
$$
x_c=\operatorname{clip}_{[0,1]}\left(\frac{v_c-v_{B,c}}{v_{W,c}-v_{B,c}}\right)
$$
:::

重みは接続ごとに入力へ掛ける係数，バイアスは各計算単位へ加える調整値である．最初の隠れ層の計算単位1個では，式[加重和](#eq:affine)のように入力と重みの積を合計する．例えば入力(0.7,0.2,0.1)，重み(0.5,−0.2,0.3)，バイアス0.01なら，合計は0.35となる．これは仕組みの説明用の値である．

::: {#eq:affine .numbered-equation}
$$z=w_Rx_R+w_Gx_G+w_Bx_B+b$$
:::

全計算単位を行列でまとめたものが式[ネットワーク](#eq:network)である．$W_1,W_2,W_3$ の大きさは順に $H\times3,H\times H,3\times H$ となる．

::: {#eq:network .numbered-equation}
$$
\begin{aligned}
\boldsymbol h_1&=\operatorname{ReLU}(W_1\boldsymbol x+\boldsymbol b_1),\\
\boldsymbol h_2&=\operatorname{ReLU}(W_2\boldsymbol h_1+\boldsymbol b_2),\\
\boldsymbol y&=W_3\boldsymbol h_2+\boldsymbol b_3
\end{aligned}
$$
:::

隠れ層には式[ReLU](#eq:relu)を用い，負の値を0にする．非線形処理を挟むことで，複数の線形変換だけでは表せない入力と出力の関係を学習できる．出力層は線形とし，評価時にクリップする．

::: {#eq:relu .numbered-equation}
$$\operatorname{ReLU}(z)=\max(0,z)$$
:::

図3はPCでの学習の流れである．予測と正解から損失を計算し，`backward()`で勾配を求め，Adamで重み・バイアスを更新する．学習時は正解RGBを255で割った尺度を使用した．

\Needspace{18\baselineskip}

```{=latex}
\noindent\begin{minipage}{\linewidth}
```

![学習と損失](figures/theory_learning.png){width=95%}

図3　学習と損失の関係

```{=latex}
\end{minipage}
```

損失は式[学習損失](#eq:loss)とした．$E_i$ は学習行iのRGB誤差の平均，$a_i>0$ はその行の重要度，$M=135$ は学習行数である．前半は行の重要度付き平均，後半はL1正則化[1]であり，重要度 $a_i$ とモデルの重み $W$ は区別する．

::: {#eq:loss .numbered-equation}
$$
\begin{aligned}
E_i&=\frac13\sum_{c\in\{R,G,B\}}\rho_\beta(\widehat y_{i,c}-t_{i,c}),\\
\mathcal L&=\frac{\sum_{i=1}^{M}a_iE_i}{\sum_{i=1}^{M}a_i}
+\lambda\sum_{\ell=1}^{3}\sum_{j,k}|(W_\ell)_{jk}|,\qquad\lambda=10^{-6}
\end{aligned}
$$
:::

RGB誤差には式[Smooth L1](#eq:smooth-l1)のSmooth L1[5]を用いた．$e$ は予測と正解の差である．小さい誤差では二乗，大きい誤差では直線的に損失が増え，一部の大きな残差の影響を抑える．$\beta=0.05$ はその切替値である．

::: {#eq:smooth-l1 .numbered-equation}
$$
\rho_\beta(e)=\begin{cases}e^2/(2\beta)&(|e|<\beta),\\|e|-\beta/2&(|e|\geq\beta),\end{cases}\qquad \beta=0.05
$$
:::

L1項は3層の重みの絶対値を足し，大きな重みへ罰則を与える．層番号を $\ell$，行列内の位置をj,kとし，バイアスは罰則の対象から除いた．小さな重みを作る方向に作用するが，重みを実際に0へ変更する処理は次節の枝刈りで行う．

## 枝刈りとCSR

枝刈り[2]では，学習後の重みへ式[枝刈り](#eq:pruning)を適用する．$\tau$ は閾値であり，絶対値が小さい重みを0にする．バイアスは変更しない．

::: {#eq:pruning .numbered-equation}
$$
w^{(\tau)}=\begin{cases}0&(|w|<\tau),\\w&(|w|\geq\tau)\end{cases}
$$
:::

図4は閾値0.05を用いた説明例である．0.03や−0.02は0になり，0.5や−0.2は残る．実験で採用した閾値は表3に示す．枝刈りの $\tau$ と，Smooth L1の $\beta$ は用途が異なる．

\Needspace{14\baselineskip}

```{=latex}
\noindent\begin{minipage}{\linewidth}
```

![枝刈りとCSR変換](figures/theory_pruning_csr.png){width=98%}

図4　枝刈りとCSR変換

```{=latex}
\end{minipage}
```

0を配列に残したままでは保存領域が必要なため，非ゼロ重みだけをCSR[3]で保存する．`values` は重みの値，`col` は元の列番号，`row` は各行の開始位置と終端を表す．図4の `row=[0,1,3]` では，行0にvaluesの0番，行1に1〜2番を使う．列番号を使えば，省略した0の位置も含めて元と同じ積和計算ができる．

CSR変換は枝刈り後の行列を表現し直す処理であり，精度への影響は枝刈り前のDenseとの比較で確認する．3層の非ゼロ重み数をKとしたとき，式[疎性](#eq:sparsity)は重みを0にした割合を表す．

::: {#eq:sparsity .numbered-equation}
$$S=1-\frac{K}{H^2+6H}$$
:::

@@HAND_CALCULATION@@

## 評価指標

出力を0〜1へクリップし，255倍して四捨五入した整数RGBで評価した．9色の補正値 $y$ と正解 $t$ から，式[MSE](#eq:mse)でMSEを求める．9色×RGBの3成分で，比較数は27個となる．

::: {#eq:mse .numbered-equation}
$$\mathrm{MSE}=\frac{1}{27}\sum_{i=1}^{9}\sum_{c\in\{R,G,B\}}(y_{i,c}-t_{i,c})^2$$
:::

説明例として，1色の正解が(255,0,0)，予測が(245,10,5)で，残り8色が一致した場合，二乗誤差の合計は225，MSEは $225/27=8.33\ldots$ となる．MSEは小さいほど正解に近い．

各回のMSEを $m_r$ とし，式[平均とSD](#eq:statistics)で10回平均と標本標準偏差（SD）を算出した．平均で誤差の大きさ，SDで読取ごとのばらつきを評価する．

::: {#eq:statistics .numbered-equation}
$$\overline m=\frac1{10}\sum_{r=1}^{10}m_r,\qquad
s_m=\sqrt{\frac1{9}\sum_{r=1}^{10}(m_r-\overline m)^2}$$
:::

MSE・標本SD・対応差は小数第2位まで表示し，第3位以降を切り捨てる．集計・増加率の計算には切り捨て前の値を使用する．

同じ読取回における差を式[対応差](#eq:paired-difference)で求める．正の値はCSRで誤差が増えたことを示す．

::: {#eq:paired-difference .numbered-equation}
$$d_r=m_{r,\mathrm{CSR}}-m_{r,\mathrm{Dense}}$$
:::

MSE増加率は式[MSE増加率](#eq:mse-change)で求め，5%以下を精度維持の条件とした．各回の割合を平均するのではなく，両方式の10回平均を比較する．

::: {#eq:mse-change .numbered-equation}
$$R_{\mathrm{MSE}}=100\left(\frac{\overline m_{\mathrm{CSR}}}{\overline m_{\mathrm{Dense}}}-1\right)\quad[\%]$$
:::

容量Bと推論時間Tの削減率は式[削減率](#eq:reductions)で算出した．いずれもDenseを基準とし，正の値が容量・時間の削減を表す．

::: {#eq:reductions .numbered-equation}
$$
R_B=100\left(1-\frac{B_{\mathrm{CSR}}}{B_{\mathrm{Dense}}}\right),\qquad
R_T=100\left(1-\frac{T_{\mathrm{CSR}}}{T_{\mathrm{Dense}}}\right)\quad[\%]
$$
:::

# 実験方法

独立変数はHと方式，従属変数はMSE・モデル配列容量・Flash・静的RAM・推論時間とした．同じHでは，元モデル・入力・RGB丸め・ボード・計時条件をそろえた．H間では追加学習前の重みも異なる．

## 装置と実行環境

| 項目 | 実施条件 |
| --- | --- |
| ボード | Arduino UNO R4 WiFi |
| 主マイコン・クロック | Renesas RA4M1，48 MHz |
| 公称Flash／SRAM | 262,144／32,768 byte |
| センサ | A0へ接続した光センサ，型番未確認 |
| LED | R:D8，G:D6，B:D7，analogWrite 255／0 |
| 読取設定 | 各チャンネル10サンプル平均，サンプル間2 ms，点灯後10 ms待機 |
| 操作 | シリアル9600 bps，外付けD2/D3ボタンは不使用 |
| PC環境 | Mac，Jupyter Notebook |
| 最終学習・集計 | クラウドPython 3.12，PyTorch 2.8.0，NumPy 2.3.5 |
| 実機コンパイラ | 7.2.1 20170904，ARM/embedded-7-branch revision 255204 |
| Arduino IDE・コア | 版番号未記録，コンパイラ文字列を保存 |

紙は手で支え，照明を同じ状態に保つ手順で測定した．照度・距離・角度の定量記録はない．

## 学習データと正解画像

学習データは実測135行（3×3分111行，6×4分24行）を使用した．3×3の教師値と評価には `result.ppm` を使用し，6×4分には配布Notebookの教師値を使用した．

各Hは同じHのV9重みから40,000 epoch追加学習した．seed42，Adam，cosine学習率0.0005→0.000005とし，色・校正グループや過去入力の整合度に基づく正の行重要度を使用した．

データ拡張として，3×3入力の50%にRGB別gain 0.75〜1.30とoffsetを加えた．灰・白には30%の確率で共通gain 0.5〜1.15を適用した．offset幅は学習入力間の距離から定め，設定と行重要度をコード・CSVへ保存した．これは学習時の変換であり，物理測定の追加ではない．

@@IMPLEMENTATION@@

## 閾値選択

学習済み重みの絶対値について，隣接する異なる値の中間を候補とし，候補数は固定しなかった．検証128入力でMSE増加5%以下かつDenseより小容量となり，学習内V8の9入力でもMSE300以下となる候補から最小容量のものを選んだ．新規10回の開始前に重み・閾値を固定した．

下表は選択に使用した入力での結果である．検証2回目には異なる6色が同じ0入力になる問題があり，選択は暫定とした．

| H | 検証Dense MSE | 検証CSR MSE | 変化率 | 学習内V8の9入力CSR MSE |
| --- | ---: | ---: | ---: | ---: |
@@VALIDATION@@

## 紙の新規10回読取

`TEACHER_FRESH_V10` で，各回に黒b・白wを登録した．各RGBの読取値を式[正規化](#eq:normalization)で正規化し，入力とした．

赤・緑・青・黄・水色・紫・黒・灰・白の順に紙を当て直し，rを各1回送信した．同じ新規入力を全8モデルへ与え，9色終了後に8個のMSEと8枚のPPMを保存した．nで次回の校正へ進み，CAPTURE=1〜10を実施した．反復単位は9色の読取1回であり，内部10サンプル平均は反復回数に数えない．

RAW・黒白校正値・正規化入力・モデル識別値・閾値・PPMを全回保存した．第2回は高MSEだったが，操作ミスの申告はなく，全10回を集計した．除外した回はない．

## 速度と容量の測定

`TEACHER_RESULT_COMPARE_V10` の `timeModel()` は，指定モデルの推論を保存済み実測9入力で100回行い，`micros()` [4]で経過時間を求める関数である．`comparison_models::predict()` で推論を呼び出し，出力の合計をチェックサムとして保持する．`runBenchmarks()` は各モデル10回のウォームアップ後に100推論×5組を計時し，Dense先行とCSR先行を交互にする．経過時間を100で割り，1推論当たりの時間を求めた．

計時区間は推論・入力選択・チェックサム加算を含み，センサ読取・シリアル出力・MSE計算を除く．新規10回の読取時間や，推論関数単体の時間とは異なる．

容量は，重み・バイアス・CSR索引の数値配列，8条件入りプログラム全体，単独モデル入りプログラム全体を分けて評価した．単独比較は `memory_probe_v10` で8条件を各1回コンパイルし，Arduino IDEのFlash・静的RAM表示を記録した．共通の初期化・推論呼出し・シリアル出力を含み，センサ読取・MSE計算を除く．単独版の実行確認，IDE・コア版とバイナリは未記録である．

@@PLAN_CHANGES@@

\Needspace{18\baselineskip}

# 結果

## 新規10回のMSE

表1　全10回のMSE集計

| H | 方式 | 平均MSE | 標本SD | 最小 | 最大 | 300以下の回数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
@@SUMMARY@@

平均MSE300以下はDenseのH60・80・100であった．CSRは全Hで平均300を超え，H100は全回で300を超えた．

\Needspace{10\baselineskip}

表2　同じ回におけるCSR−Denseの差

| H | 平均差 | 差の標本SD | 平均MSEの相対増加 |
| --- | ---: | ---: | ---: |
@@PAIRED@@

全10回・全HでCSRのMSEがDenseを上回り，平均MSEの増加率は41.46〜150.33%となった．精度維持の条件である5%以下は達成しなかった．

\Needspace{28\baselineskip}

```{=latex}
\noindent\begin{minipage}{\linewidth}
```

![新規10回のMSE](figures/physical_mse10.png)

図5　各回のMSE（縦軸は対数目盛　破線は目標300）

```{=latex}
\end{minipage}
```

## モデル配列容量

\Needspace{10\baselineskip}

表3　枝刈り閾値と配列容量

| H | 閾値 | NNZ | 疎性 | Dense byte | CSR byte | 削減率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
@@PAYLOAD@@

NNZは非ゼロ重み数，疎性は重みを0にした割合を示す．全HでCSR索引を含む配列容量が減少した．最小容量はH40の2,854 byte，最大削減率はH100の69.70%だった．

## 実機推論時間

\Needspace{10\baselineskip}

表4　固定入力での推論時間（µs/推論）

| H | Dense 平均±SD | CSR 平均±SD | 時間短縮率 |
| --- | ---: | ---: | ---: |
@@TIMING@@

全Hで28.97〜44.01%短縮した．表4の±は100推論の平均時間5組の標本SDであり，表1の読取10回のMSEのSDとは異なる．

\Needspace{18\baselineskip}

```{=latex}
\noindent\begin{minipage}{\linewidth}
```

![容量と実機推論時間](figures/capacity_and_time.png)

図6　モデル配列容量と推論時間（誤差棒は5組の標本SD）

```{=latex}
\end{minipage}
```

\Needspace{14\baselineskip}

## Flash・RAMと動作確認

\Needspace{10\baselineskip}

表5　8条件入りプログラム全体のArduino IDE使用量（byte）

| プログラム | Flash | 静的RAM | 静的割当後の残りRAM |
| --- | ---: | ---: | ---: |
| 保存入力再生・速度版 | 187,888 | 6,740 | 26,028 |
@@MEMORY@@

\Needspace{10\baselineskip}

表6　単独モデル入りプログラムの使用量（byte）

| H | Dense Flash | CSR Flash | Flash削減量 | 削減率 | 両方式の静的RAM |
| --- | ---: | ---: | ---: | ---: | ---: |
@@SINGLE_MEMORY@@

条件名付きIDE表示に基づき，全HでFlashが減少した．H100では30,128 byte（31.26%）削減した．静的RAMは全8条件で6,764 byte，静的割当後の残りは26,004 byteだった．

\Needspace{19\baselineskip}

```{=latex}
\noindent\begin{minipage}{\linewidth}
```

![単独モデルのFlash](figures/single_model_flash.png)

図7　単独モデルのFlash使用量

```{=latex}
\end{minipage}
```

Flash全体はArduinoコア等を含み，モデル配列だけの削減率とは異なる．モデルはconst配列であり，関数内の中間配列も静的RAMには含まれない．残り26,004 byteはローカル変数の実使用量ではない．中間配列はfloat[100]を2本，宣言上800 byteだが，実行時スタックの最大量は未測定である．

保存入力版の実機自己検査とCSR索引検査はPASSした．新規10回の全80PPM，計2,160 RGB成分は固定V10のPC再計算と一致した．80個のMSEは先生の元の計算HTMLとも表示桁の範囲で一致した．

# 考察

## 容量・速度と精度の両立

検証128入力ではMSE増加5%以下を満たしたが，新規10回では満たさなかった．入力が同じ0へ潰れる問題や読取状態の変動が考えられるものの，支配的な要因は特定できない．今回の閾値では，枝刈りによる出力への影響を十分小さくできなかった．

@@QUANTITATIVE@@

## 第2回とばらつき

第2回は全モデルでMSEが大きく，平均とSDに強く影響した．H60のDenseは最小@@H60_DENSE_MIN@@，最大@@H60_DENSE_MAX@@，平均@@H60_DENSE_MEAN@@，標本SD@@H60_DENSE_SD@@だった．

青・水色などのRAW・正規化入力も他の回と異なった．PCと実機の出力は一致しているため，MSE表示だけの差ではない．校正・照射・紙の位置・センサ応答が原因候補だが，照度や距離の記録がなく断定できない．操作ミスの申告もなかったため第2回を保持した．

## 次元数と評価の限界

Denseの平均MSEはH80で最小だったが，Hの増加に対して単調には改善しなかった．Hごとに初期重みも異なり，H単独の効果とは断定できない．H40のCSRは最小容量・最短時間，H100は最大削減率・時間短縮率だったが，精度維持を同時に満たした条件はなかった．

測定は手持ちの同一チャートを続けて10回読んだものであり，距離・角度・照度は定量統制していない．独立最終チャート，H50，各方式の搭載限界，最大スタック使用量は未測定である．学習損失，学習行を再入力したMSE，新規読取のMSEは分けて解釈する．

@@FUTURE_WORK@@

# 結論

H40・60・80・100の枝刈り＋CSRは，Denseに対しモデル配列容量を60.51〜69.70%，固定入力での実機推論時間を28.97〜44.01%，単独ビルドのFlashを7.94〜31.26%削減した．静的RAMは全条件で6,764 byteと変わらなかった．

一方，新規10回の平均MSEは全Hで増加し，MSE増加5%以下とCSRの平均MSE300以下は達成しなかった．容量・速度の改善は確認したが，精度維持との両立には，入力の再現性，閾値選択，枝刈り後の再学習を見直し，独立チャートで検証する必要がある．

# 参考文献

[1] R. Tibshirani， “Regression Shrinkage and Selection via the Lasso，” Journal of the Royal Statistical Society: Series B， 58(1)， pp. 267–288， 1996．

[2] S. Han， J. Pool， J. Tran， and W. J. Dally， “Learning Both Weights and Connections for Efficient Neural Networks，” Advances in Neural Information Processing Systems 28， pp. 1135–1143， 2015．

[3] N. Bell and M. Garland， “Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented Processors，” Proceedings of SC '09， Article 18， pp. 1–11， 2009．

[4] Arduino， “micros()，” Arduino Documentation， <https://docs.arduino.cc/language-reference/en/functions/time/micros/>．

[5] PyTorch Contributors， “SmoothL1Loss，” PyTorch Documentation， <https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html>．

# 付録：各回の実測値と再現資料

\Needspace{16\baselineskip}

## 全10回のDense MSE

| 回 | H40 | H60 | H80 | H100 |
| --- | ---: | ---: | ---: | ---: |
@@RAW_DENSE@@

\Needspace{16\baselineskip}

## 全10回の枝刈り＋CSR MSE

| 回 | H40 | H60 | H80 | H100 |
| --- | ---: | ---: | ---: | ---: |
@@RAW_CSR@@

## 学習135行の重みなしMSE

全行の重要度を等しくしたMSEを示す．学習損失や新規10回のMSEとは異なる．

| H | 学習135行のMSE | うち3×3の111行のMSE |
| --- | ---: | ---: |
@@TRAINING@@

使用コード・集計CSV・元ログと再計算手順は同梱READMEに，各ファイルのSHA256は `manifest.json` に示す．

@@CODE_APPENDIX@@
