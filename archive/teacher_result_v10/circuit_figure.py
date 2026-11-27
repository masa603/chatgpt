"""Publish the user's original circuit image without redrawing or editing it."""

import hashlib
import json
from pathlib import Path
import shutil

import fitz
from PIL import Image

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build_circuit_figure(folder, source):
    folder, source = Path(folder), Path(source)
    folder.mkdir(parents=True, exist_ok=True)
    original = HERE/'reference/circuit_original.png'
    provenance = json.loads((HERE/'reference/circuit_source.json').read_text())
    raw = original.read_bytes()
    assert sha(raw) == provenance['image_sha256']
    with Image.open(original) as image:
        width, height = image.size
        rgb = image.convert('RGB').tobytes()
    assert (width, height) == (provenance['image_width'], provenance['image_height'])

    # Retain the original PNG bytes, including its color profile and margins.
    shutil.copyfile(original, folder/'circuit.png')
    assert (folder/'circuit.png').read_bytes() == raw

    # Embed the full image losslessly in a PDF with the same aspect ratio.
    document = fitz.open()
    page = document.new_page(width=720, height=720*height/width)
    xref = page.insert_image(page.rect, stream=raw)
    pixels = fitz.Pixmap(document, xref)
    assert (pixels.width, pixels.height, pixels.n) == (width, height, 3)
    assert pixels.samples == rgb
    document.save(folder/'circuit.pdf', garbage=4, deflate=True)
    document.close()

    record = {'role':'USER_SUPPLIED_ORIGINAL_CIRCUIT_IMAGE',
              'source':'reference/circuit_original.png',
              'image_sha256':sha(raw), 'width':width, 'height':height,
              'original_png_byte_identical':True, 'image_edited':False,
              'cropped':False, 'redrawn':False,
              'pdf_embedding':'LOSSLESS_RGB_PIXELS; FULL_IMAGE; ORIGINAL_ASPECT_RATIO',
              'rgb_pixels_sha256':sha(rgb),
              'frozen_firmware_sha256':sha(source.read_bytes()),
              'source_provenance':provenance}
    (folder/'circuit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')

    # Expose only the chosen image, removing obsolete redraw exports.
    for suffix in ['png','pdf','svg','json']:
        (folder/('circuit_two_buttons.'+suffix)).unlink(missing_ok=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    build_circuit_figure(args.output, args.source)
