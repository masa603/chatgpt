"""TA向け：Enterで段階を進める、実モデルの枝刈り・CSR変換の実演。"""

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np

from threshold_tools import load_model, prune, save_csr_npz, storage


DEFAULT_THRESHOLD = 0.2449173554778099
REFERENCE_MODEL_SHA256 = "429d013f77f02c7aaf64cc6cba5020e23d2c7641ba1bdc69f4f162d352cdb658"


def dense_forward(layers, sample):
    x = np.asarray(sample, dtype=np.float32)
    for layer, (weight, bias) in enumerate(layers):
        output = bias.copy()
        for column in range(weight.shape[1]):
            output += weight[:, column] * x[column]
        x = np.maximum(output, np.float32(0)) if layer < 2 else output
    return x


def csr_forward(arrays, sample):
    x = np.asarray(sample, dtype=np.float32)
    for layer in (1, 2, 3):
        values = arrays[f"values_{layer}"]
        columns = arrays[f"col_idx_{layer}"]
        pointers = arrays[f"row_ptr_{layer}"]
        output = arrays[f"bias_{layer}"].copy()
        for row in range(len(output)):
            for position in range(int(pointers[row]), int(pointers[row + 1])):
                output[row] += values[position] * x[columns[position]]
        x = np.maximum(output, np.float32(0)) if layer < 3 else output
    return x


def run_demo(header, threshold, output_dir, pause):
    lines = []

    def show(value=""):
        text = str(value)
        print(text)
        lines.append(text)

    def advance(label):
        if pause:
            input(f"\nEnterで「{label}」へ進みます…")
        show()

    def matrix(values):
        return np.array2string(values, max_line_width=110,
                               formatter={"float_kind": lambda x: f"{float(x): .6g}"})

    layers = load_model(header)
    before = storage(layers)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_rows = min(8, len(layers[0][1]))
    show("枝刈り＋CSRの実演（実際のモデル重みを使用）")
    show(f"モデル: {header}")
    show(f"隠れ層 H={len(layers[0][1])}、閾値={threshold}")
    show("この実演は、指定した閾値による変換と計算の一致を示します。")
    show("既存H=40モデルのデフォルト閾値は暫定値です。色精度の評価は別に行います。")
    show()
    show("【1/4】枝刈り前")
    show(f"重み数: {sum(w.size for w, _ in layers)}、非ゼロ数: {before['nonzero']}")
    show(f"Denseモデル配列容量: {before['dense_bytes']} byte")
    show(f"第1層の先頭{preview_rows}行（3列）:")
    show(matrix(layers[0][0][:preview_rows]))
    np.savetxt(output_dir / "layer1_before.txt", layers[0][0], fmt="%.9g")

    advance("枝刈り")
    show("【2/4】絶対値が閾値より小さい重みを0にする")
    sparse = prune(layers, threshold)  # 実際の処理：threshold_tools.py の prune()
    after = storage(sparse)
    for i, ((old, bias), (weight, new_bias)) in enumerate(zip(layers, sparse), start=1):
        if not np.array_equal(bias, new_bias):
            raise AssertionError("Bias was changed.")
        show(f"第{i}層 {old.shape}: 非ゼロ {np.count_nonzero(old)} → {np.count_nonzero(weight)}")
    show(f"第1層の先頭{preview_rows}行:")
    show(matrix(sparse[0][0][:preview_rows]))
    show("バイアスは変更せず、重みとバイアスはFP32のままです。")
    show(f"0化後もDense配列の容量は {after['dense_bytes']} byteです。")
    np.savetxt(output_dir / "layer1_pruned.txt", sparse[0][0], fmt="%.9g")

    advance("CSR形式への変換")
    show("【3/4】非ゼロ値・列番号・行ポインタの3配列へ変換する")
    archive = output_dir / "demo_csr.npz"
    payload_bytes = save_csr_npz(sparse, archive)  # 実際の処理：save_csr_npz()
    with np.load(archive, allow_pickle=False) as packed:
        arrays = {name: packed[name].copy() for name in packed.files}
    end = int(arrays["row_ptr_1"][preview_rows])
    show(f"以下は第1層の先頭{preview_rows}行に対応する部分です。")
    show("values  (float32) = " + matrix(arrays["values_1"][:end]))
    show("col_idx (uint8)   = " + str(arrays["col_idx_1"][:end]))
    show("row_ptr (uint16)  = " + str(arrays["row_ptr_1"][:preview_rows + 1]))
    show("列番号は0始まりです。row_ptrで隣り合う値が同じ行には、非ゼロ重みがありません。")
    show(f"全3層のCSRモデル配列容量: {payload_bytes} byte")
    show(f"Denseからの配列データ削減率: {after['saving_pct']:.2f}%")
    show("容量にはFP32バイアスとCSR索引を含みます。NPZファイル自体のサイズとは別です。")
    all_arrays = []
    for name, values in arrays.items():
        all_arrays.append(f"{name} ({values.dtype})\n{matrix(values)}\n")
    (output_dir / "csr_arrays.txt").write_text("\n".join(all_arrays), encoding="utf-8")

    advance("復元と推論結果の確認")
    show("【4/4】CSRから復元し、枝刈り後Denseと一致するか確認する")
    for layer, (weight, bias) in enumerate(sparse, start=1):
        restored = np.zeros_like(weight)
        columns, pointers = arrays[f"col_idx_{layer}"], arrays[f"row_ptr_{layer}"]
        for row in range(len(bias)):
            start, stop = map(int, pointers[row:row + 2])
            restored[row, columns[start:stop]] = arrays[f"values_{layer}"][start:stop]
        if not np.array_equal(restored, weight):
            raise AssertionError(f"Layer {layer}: CSR reconstruction failed.")
    show("3層の重み行列の復元: PASS")
    sample = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    dense_result = dense_forward(sparse, sample)
    csr_result = csr_forward(arrays, sample)
    np.testing.assert_allclose(csr_result, dense_result, rtol=1e-5, atol=1e-5)
    show("固定入力: " + matrix(sample))
    show("NN出力（クリップ・0〜255変換前）:")
    show("  元Dense       = " + matrix(dense_forward(layers, sample)))
    show("  枝刈り後Dense = " + matrix(dense_result))
    show("  CSR           = " + matrix(csr_result))
    show("枝刈り後DenseとCSRの出力一致: PASS")
    show(f"最大絶対差: {float(np.max(np.abs(dense_result - csr_result))):.9g}")
    show(f"保存先: {output_dir}")
    (output_dir / "demo_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"sparse": sparse, "arrays": arrays, "bytes": payload_bytes}


def main():
    here = Path(__file__).resolve().parent
    choices = [here / "model_parameters.h", here.parent / "hdjsk/model_parameters.h"]
    default_header = next((path for path in choices if path.exists()), choices[0])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, default=default_header)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=here / "demo_output")
    parser.add_argument("--no-pause", action="store_true", help="Enter待ちをせずに全工程を実行する")
    args = parser.parse_args()
    if not args.header.exists():
        parser.error("元のDense用 model_parameters.h を、このPythonファイルと同じフォルダへ置いてください。")
    if args.threshold is None:
        if hashlib.sha256(args.header.read_bytes()).hexdigest() != REFERENCE_MODEL_SHA256:
            parser.error("元モデルが異なります。実演用の閾値を --threshold で指定してください。")
        threshold = DEFAULT_THRESHOLD
    else:
        threshold = args.threshold
    if not np.isfinite(threshold) or threshold < 0:
        parser.error("閾値には0以上の有限値を指定してください。")
    run_demo(args.header, threshold, args.output_dir, pause=not args.no_pause and sys.stdin.isatty())


if __name__ == "__main__":
    main()
