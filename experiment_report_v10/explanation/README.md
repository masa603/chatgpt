# レポートを理解するための解説

[解説PDFを開く](guide.pdf)

本人が式・コード・実験内容を理解するための資料です。提出用の本文は [report.pdf](../report.pdf) にあります。
学習と実測、枝刈りとCSR、各画像のMSEと10回平均、モデル配列とFlash・静的RAMを順番に説明しています。
コードは使用済みプログラムから抜粋し、平均MSEは元の実測CSVから入れています。

説明の原稿は [guide.md](guide.md)、LaTeXソースは [guide.tex](guide.tex)、照合記録は [verification.json](verification.json) です。
再作成はsource_snapshotを作業フォルダとして `python teacher_result_v10/build_explanation_guide.py` で行えます。
pandoc、XeLaTeXとNoto CJKフォントが必要です。測定や再学習は行いません。
