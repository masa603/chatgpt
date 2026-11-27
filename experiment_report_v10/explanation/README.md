# レポートを理解するための解説

[解説PDFを開く](guide.pdf)

[電車で読むA5解説PDF（23ページ）](commute_guide.pdf)

[完成版を口頭で説明するためのPDF](oral_guide.pdf)

口頭説明用は，実験の回数，式17個，使用コード，結果と考察，想定質問を32項目で説明しています．各項目の冒頭を声に出し，続く根拠を確認する形で練習できます．原稿は [oral_guide.md](oral_guide.md)，照合記録は [oral_verification.json](oral_verification.json) です．`python teacher_result_v10/build_oral_guide.py` で再作成できます．

電車用は文字を大きくし，1つの話題を1ページで説明しています．実験の回数・結果，式の計算例，実際に使ったコードの読み方と回路図を収録し，保存後はオフラインで読めます．

本人が式・コード・実験内容を理解するための資料です。提出用の本文は [report.pdf](../report.pdf) にあります。
学習と実測、枝刈りとCSR、各画像のMSEと10回平均、モデル配列とFlash・静的RAMを順番に説明しています。
コードは使用済みプログラムから抜粋し、平均MSEは元の実測CSVから入れています。

説明の原稿は [guide.md](guide.md)、LaTeXソースは [guide.tex](guide.tex)、照合記録は [verification.json](verification.json) です。
再作成はsource_snapshotを作業フォルダとして `python teacher_result_v10/build_explanation_guide.py` で行えます。
pandoc、XeLaTeXとNoto CJKフォントが必要です。測定や再学習は行いません。

電車用の原稿は [commute_guide.md](commute_guide.md)，照合記録は [commute_verification.json](commute_verification.json) です．同じ作業ディレクトリから `python teacher_result_v10/build_commute_guide.py` で再作成できます．
