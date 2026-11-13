---
title: "枝刈り・CSRによる色補正モデルの圧縮"
subtitle: "アジャイルワーク2　実験報告書"
author: "@@ID@@　@@NAME@@"
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

ネットワークは3−H−H−3の全結合構造とした．入力を $\boldsymbol x$，重みを $W$，バイアスを $\boldsymbol b$ とすると，

$$
\begin{aligned}
\boldsymbol h_1&=\operatorname{ReLU}(W_1\boldsymbol x+\boldsymbol b_1),\\
\boldsymbol h_2&=\operatorname{ReLU}(W_2\boldsymbol h_1+\boldsymbol b_2),\\
\boldsymbol y&=W_3\boldsymbol h_2+\boldsymbol b_3.
\end{aligned}
$$

隠れ層にReLUを用い，出力層は線形とした．学習には，重み付きSmooth L1損失[5]と，バイアスを除く重みへのL1正則化[1]を用いた．

$$
\mathcal L=\frac{1}{\sum_{i=1}^{M}a_i}
\sum_{i=1}^{M}\frac{a_i}{3}\sum_{c\in\{R,G,B\}}
\ell_\beta(\widehat y_{i,c}-t_{i,c})
+10^{-6}\sum_{\ell=1}^{3}\sum_{j,k}|(W_\ell)_{jk}|,
$$

$$
\ell_\beta(e)=\begin{cases}e^2/(2\beta)&(|e|<\beta),\\|e|-\beta/2&(|e|\geq\beta),\end{cases}\qquad \beta=0.05
$$

$M=135$ は学習行数，$a_i>0$ は行ごとの重要度，$\widehat y$ は予測，$t$ は正解，$c$ はRGB成分を表す．学習時の尺度は0〜1とした．$\ell$ は層番号，jとkは行列内の位置である．L1項は全3層の重みの絶対値を合計し，小さい重みを作る方向に作用する．学習損失と，評価時の整数RGBのMSEは区別する．

## 枝刈りとCSR

枝刈り[2]では $|w|<\tau$ の重みを0にし，バイアスは保持した．残った重みを，値・列番号・行開始位置の3配列からなるCSR[3]へ変換した．保存・計算対象を非ゼロ重みに限定する．3層の非ゼロ重み数をKとすると，疎性Sは次式である．

$$S=1-\frac{K}{H^2+6H}$$

@@HAND_CALCULATION@@

## 評価指標

出力を0〜1へクリップし，255倍して四捨五入した整数RGBで評価した．9色の補正値 $y$ と正解 $t$ から，MSEを次式で求めた．

$$\mathrm{MSE}=\frac{1}{27}\sum_{i=1}^{9}\sum_{c\in\{R,G,B\}}(y_{i,c}-t_{i,c})^2$$

\Needspace{6\baselineskip}

各回のMSEを $m_r$ とし，10回平均と標本標準偏差（SD）を算出した．

$$\overline m=\frac1{10}\sum_{r=1}^{10}m_r,\qquad
s_m=\sqrt{\frac1{9}\sum_{r=1}^{10}(m_r-\overline m)^2}$$

MSE・標本SD・対応差は小数第2位まで表示し，第3位以降を切り捨てる．集計・増加率の計算には切り捨て前の値を使用する．

同じ回の差を $d_r=m_{r,\mathrm{CSR}}-m_{r,\mathrm{Dense}}$ とした．MSE増加率は $100(\overline m_{\mathrm{CSR}}/\overline m_{\mathrm{Dense}}-1)$ とし，各回の割合の平均は使わない．容量Bと時間Tの削減率は，それぞれ $100(1-B_{\mathrm{CSR}}/B_{\mathrm{Dense}})$，$100(1-T_{\mathrm{CSR}}/T_{\mathrm{Dense}})$ で求めた．

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

実測135行を全て保持し，配布Notebookの見本入力は実測として追加しなかった．正解画像の取り違えを修正し，最終V10では3×3分111行の教師値と評価を，先生指定の元の `result.ppm` へ統一した．灰色の正解は(127,127,127)である．6×4分24行は元のNotebookの教師値を保持したが，別の先生指定PPMとの照合はできていない．旧CMYK基準のMSEは結果へ混在させない．

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

`TEACHER_FRESH_V10` で，各回に黒b・白wを登録した．各RGBの読取値vから $(v-b)/(w-b)$ を求め，0〜1へクリップして入力とした．

赤・緑・青・黄・水色・紫・黒・灰・白の順に紙を当て直し，rを各1回送信した．同じ新規入力を全8モデルへ与え，9色終了後に8個のMSEと8枚のPPMを保存した．nで次回の校正へ進み，CAPTURE=1〜10を実施した．反復単位は9色の読取1回であり，内部10サンプル平均は反復回数に数えない．

RAW・黒白校正値・正規化入力・モデル識別値・閾値・PPMを全回保存した．第2回は高MSEだったが，操作ミスの申告はなく，全10回を集計した．除外した回はない．

## 速度と容量の測定

`TEACHER_RESULT_COMPARE_V10` で保存済み実測9入力を巡回した．各モデル10回のウォームアップ後，100推論×5組を計時し，Dense先行とCSR先行を交互にした．`micros()` [4]の差を100で割り，1推論当たりの時間を求めた．

\Needspace{16\baselineskip}

```cpp
@@TIMER_SOURCE@@
```

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

図2　各回のMSE（縦軸は対数目盛　破線は目標300）

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

図3　モデル配列容量と推論時間（誤差棒は5組の標本SD）

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

図4　単独モデルのFlash使用量

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
