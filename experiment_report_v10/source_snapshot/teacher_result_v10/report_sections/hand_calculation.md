## メモリ容量の手計算

Denseの重み数は $N_w=3H+H^2+3H$，バイアス数は $N_b=2H+3$ である．FP32は1個4 byteなので，

$$B_{\mathrm{Dense}}=4(N_w+N_b)=4(H^2+8H+3)\quad[\mathrm{byte}]$$

となる．CSRは非ゼロ重みK個をFP32，列番号をuint8，行開始位置をuint16で保存する．行位置は3層の行数に各終端を加えた $2H+6$ 個となる．バイアスを全て残すと，

$$
\begin{aligned}
B_{\mathrm{CSR}}&=4K+K+2(2H+6)+4(2H+3)\\
&=5K+12H+24\quad[\mathrm{byte}]
\end{aligned}
$$

H60では重み3,960個，バイアス123個，枝刈り後K=1,141個なので，

$$
\begin{aligned}
B_{\mathrm{Dense}}&=4(60^2+8\cdot60+3)=16,332,\\
B_{\mathrm{CSR}}&=4,564+1,141+252+492=6,449,\\
R_B&=100(1-6,449/16,332)=60.51\%
\end{aligned}
$$

\Needspace{10\baselineskip}

全HのCSR配列の内訳を示す．単位はbyteである．

| H | 非ゼロ重み | 列番号 | 行位置 | バイアス | 合計byte |
| --- | ---: | ---: | ---: | ---: | ---: |
@@MEMORY_COMPONENT_ROWS@@

配列容量が減る条件は $5K<4H^2+20H-12$ で，採用した4条件は全て満たす．この計算は数値配列のみを対象とし，構造体・アラインメント・Arduinoコア等を含むFlash全体はIDE表示で別に評価した．
