"""Archive actual device replay and reconstruct rounding-stable RGB images."""

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/chatgpt-experiment-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/chatgpt-experiment-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiment_today"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pruning"))
from threshold_tools import load_model, prune
from export_csr_arduino import dense_layer_references


def save_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main():
    raw_path = HERE / "hardware/latest_replay_01.txt"
    text = raw_path.read_text()
    assert "FIRMWARE=PRECISION_SCAN_V2" in text and "OUTPUT CHECK: PASS" in text
    assert "SCOPE=DEVICE_REPLAY_ON_SAVED_INPUTS; NOT_A_NEW_SENSOR_CAPTURE" in text
    manifest = json.loads((HERE / "latest_replay_manifest.json").read_text())
    source_path = BASE / manifest["source_log"]
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert source_hash == manifest["source_log_sha256"] == re.search(r"^SOURCE_LOG_SHA256=(.+)$",text,re.M)[1]
    source_meta = json.loads((source_path.parent / "summary.json").read_text())
    with (source_path.parent / "inputs.csv").open(newline="") as f:
        data = list(csv.DictReader(f))
    raw = np.array([[r[f"raw_{c}"] for c in "rgb"] for r in data], np.float32)
    black, white = np.array(source_meta["black_rgb"],np.float32), np.array(source_meta["white_rgb"],np.float32)
    inputs = np.clip((raw-black)/(white-black),0,1)
    assert hashlib.sha256(inputs.astype('<f4').tobytes()).hexdigest() == manifest['input_array_sha256_float32_le']
    targets = np.array([[r[f"target_{c}8"] for c in "rgb"] for r in data],np.int32)
    lines=text.splitlines(); index=lines.index("replay_dataset,H,format,colors,mse")
    device_rows=list(csv.DictReader(io.StringIO("\n".join(lines[index:index+7]))))
    assert [(int(r["H"]),r["format"]) for r in device_rows] == [(h,m) for h in (40,60,100) for m in ("DENSE","CSR")]
    model_meta=re.findall(r"H=(\d+)\nTHRESHOLD=([^\n]+)\nMODEL_SOURCE_SHA256=([^\n]+)",text)
    assert [int(h) for h,_,_ in model_meta] == [40,60,100]
    for h,threshold,model_hash in model_meta:
        selection=json.loads((HERE / f"selection/h{h}/summary.json").read_text())
        assert float(threshold)==selection["selected"]["threshold"] and model_hash==selection["model_sha256"]
    # Re-parse the exact PC reference floats embedded in the actually supplied firmware.
    header=(HERE / "precision_scan/precision_models.h").read_text()
    block=re.search(r"const float EXPECTED\[54\]\[3\] = \{(.*?)\n\};",header,re.S)[1]
    values=re.findall(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)f",block)
    embedded=np.array(values,np.float32).reshape(6,9,3)
    printed_error=float(re.search(r"^MAX ABS ERROR=([^\n]+)$",text,re.M)[1])
    error_bound=printed_error+1e-8  # covers positive decimal-print rounding at eight places
    images=HERE / "replay_images"; images.mkdir(exist_ok=True)
    all_rgb=[]; patch_rows=[]; reports=[]; reconstructed=[]
    for model_index, row in enumerate(device_rows):
        h,mode=int(row["H"]),row["format"]
        selection=json.loads((HERE / f"selection/h{h}/summary.json").read_text())
        dense=load_model(ROOT / f"retraining/results/h{h}/model_parameters.h")
        model=dense if mode=="DENSE" else prune(dense,selection["selected"]["threshold"])
        output=dense_layer_references(model,inputs)[-1]
        assert np.array_equal(output,embedded[model_index])
        rgb_float=np.clip(output.astype(float),0,1)*255
        margin=float(np.min(np.abs((rgb_float-np.floor(rgb_float))-.5))/255)
        assert margin > error_bound, "Printed device error does not certify these rounded pixel values."
        pixels=np.floor(rgb_float+.5).astype(np.uint8)
        mse=float(np.mean((pixels.astype(float)-targets)**2))
        assert abs(mse-float(row["mse"]))<1e-6
        filename=f"h{h}_{mode.lower()}"
        ppm="P3\n3 3\n255\n"+"\n".join(" ".join(map(str,pixels[r*3:(r+1)*3].ravel())) for r in range(3))+"\n"
        (images/f"{filename}.ppm").write_text(ppm)
        enlarged=np.repeat(np.repeat(pixels.reshape(3,3,3),120,axis=0),120,axis=1)
        plt.imsave(images/f"{filename}.png",enlarged)
        reconstructed.append(pixels.reshape(3,3,3))
        reports.append({"hidden_size":h,"format":mode,"colors":9,"mse_device_reported":float(row["mse"]),
                        "mse_recomputed_from_rgb":mse,"minimum_normalized_rounding_margin":margin,
                        "max_device_error_bound_including_print_rounding":error_bound,
                        "rgb8_equality_certificate":"PASS: all rounding thresholds farther than measured error bound",
                        "ppm":f"{filename}.ppm","png":f"{filename}.png"})
        for i,pixel in enumerate(pixels):
            all_rgb.append({"pixel":i+1,"color_id":data[i]["color_id"],"hidden_size":h,"format":mode,
                            "red":int(pixel[0]),"green":int(pixel[1]),"blue":int(pixel[2])})
            patch_rows.append({**all_rgb[-1],"mse_per_color":float(np.mean((pixel.astype(float)-targets[i])**2))})
    wide=[]
    for i in range(9):
        row={"pixel":i+1,"color_id":data[i]["color_id"]}
        for h in (40,60,100):
            for mode in ("DENSE","CSR"):
                p=next(p for p in all_rgb if p["pixel"]==i+1 and p["hidden_size"]==h and p["format"]==mode)
                row[f"h{h}_{mode.lower()}_rgb"]=f"{p['red']} {p['green']} {p['blue']}"
        wide.append(row)
    save_csv(images/"rgb9_all_conditions.csv",wide)
    save_csv(images/"rgb9_long.csv",all_rgb)
    save_csv(images/"per_color_mse.csv",patch_rows)
    save_csv(images/"mse_summary.csv",reports)
    comparison=[]
    for h in (40,60,100):
        dense,csr=[r["mse_recomputed_from_rgb"] for r in reports if r["hidden_size"]==h]
        comparison.append({"hidden_size":h,"dense_mse":dense,"csr_mse":csr,"csr_change_percent":(csr/dense-1)*100})
    ref=targets.astype(np.uint8).reshape(3,3,3)
    plt.imsave(images/"reference_recovered.png",np.repeat(np.repeat(ref,120,axis=0),120,axis=1))
    fig,axes=plt.subplots(2,4,figsize=(10.5,5.5))
    axes[0,0].imshow(ref,interpolation="nearest");axes[0,0].set_title("Reference (recovered)")
    axes[1,0].text(.05,.5,"Saved sensor input replay\nNew precision candidates\nRGB8 reconstructed and\nrounding-stability checked\nNot a new sensor capture\nNot an independent final test",fontsize=9,va="center")
    for m,image in enumerate(reconstructed):
        ax=axes[m%2,1+m//2];r=reports[m];ax.imshow(image,interpolation="nearest")
        ax.set_title(f"H={r['hidden_size']} {r['format']}\nMSE={r['mse_device_reported']:.2f}")
    for ax in axes.flat:ax.set_axis_off()
    fig.tight_layout();fig.savefig(images/"comparison.png",dpi=160);fig.savefig(images/"comparison.pdf");plt.close(fig)
    summary={"received_date_japan":"2026-11-06","execution_timestamp":None,"firmware":"PRECISION_SCAN_V2",
             "source_sensor_log_sha256":source_hash,"scope":"Actual Arduino inference on archived physical inputs, not new acquisition or independent final evaluation.",
             "output_check":"PASS","reported_max_abs_error":printed_error,"pixel_values":"Reconstructed from exact embedded FP32 references; observed error bound certifies identical RGB8 rounding for all 162 components.",
             "models":[{"hidden_size":int(h),"threshold":float(t),"model_sha256":s} for h,t,s in model_meta],
             "results":reports,"paired_comparisons":comparison,"raw_replay_log_sha256":hashlib.sha256(raw_path.read_bytes()).hexdigest()}
    raw_path.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    manifest['device_replay'] = 'RECEIVED_PASS'
    manifest['device_replay_record'] = 'hardware/latest_replay_01.json'
    for row in manifest['pc_results']:
        row['role'] = 'PC_REFERENCE; matching actual device replay recorded separately in hardware/latest_replay_01.json'
    (HERE/'latest_replay_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(comparison));print('PASS: device MSEs, model/source hashes, exact firmware reference floats and all 162 rounding-stable RGB8 components.')


if __name__ == "__main__":
    main()
