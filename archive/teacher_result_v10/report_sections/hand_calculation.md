## メモリ容量の手計算

図2の各接続に重みが1個ずつあるため，Denseの重み数は $N_w=3H+H^2+3H$ となる．各計算単位のバイアスは $N_b=2H+3$ 個である．FP32は1個4 byteなので，式[Dense容量](#eq:dense-bytes)で配列容量を求める．

::: {#eq:dense-bytes .numbered-equation}
$$B_{\mathrm{Dense}}=4(N_w+N_b)=4(H^2+8H+3)\quad[\mathrm{byte}]$$
:::

CSRは非ゼロ重みK個をFP32，列番号をuint8，行開始位置をuint16で保存する．容量はそれぞれ4K，K，$2(2H+6)$ byteとなる．行位置の個数は各層の行数に終端を1個ずつ足した $(H+1)+(H+1)+(3+1)=2H+6$ 個である．バイアスを全て残すと，総容量は式[CSR容量](#eq:csr-bytes)となる．

::: {#eq:csr-bytes .numbered-equation}
$$
\begin{aligned}
B_{\mathrm{CSR}}&=4K+K+2(2H+6)+4(2H+3)\\
&=5K+12H+24\quad[\mathrm{byte}]
\end{aligned}
$$
:::

H60では重み3,960個，バイアス123個，枝刈り後K=1,141個なので，式[H60の容量](#eq:h60-capacity)のように計算できる．CSR行位置は61＋61＋4＝126個で252 byte，バイアスは123個で492 byteである．

::: {#eq:h60-capacity .numbered-equation}
$$
\begin{aligned}
B_{\mathrm{Dense}}&=4(60^2+8\cdot60+3)=16,332,\\
B_{\mathrm{CSR}}&=4,564+1,141+252+492=6,449,\\
R_B&=100(1-6,449/16,332)=60.51\%
\end{aligned}
$$
:::

\Needspace{10\baselineskip}

全HのCSR配列の内訳を示す．単位はbyteである．

| H | 非ゼロ重み | 列番号 | 行位置 | バイアス | 合計byte |
| --- | ---: | ---: | ---: | ---: | ---: |
@@MEMORY_COMPONENT_ROWS@@

CSRでは位置情報も保存するため，非ゼロ重みが多いと容量が減らない．$B_{\mathrm{CSR}}<B_{\mathrm{Dense}}$ を整理した式[容量削減の条件](#eq:capacity-condition)を，採用した4条件は全て満たす．

::: {#eq:capacity-condition .numbered-equation}
$$5K<4H^2+20H-12$$
:::

この計算は数値配列のみを対象とし，構造体・アラインメント・Arduinoコア等を含むFlash全体はIDE表示で別に評価した．
