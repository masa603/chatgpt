// Execute the supplied calculator's original parser and MSE function unchanged.
// Canvas pixels use Uint8ClampedArray, as in a browser ImageData object.
const fs = require('node:fs');
const vm = require('node:vm');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');

function sha256(text) {
  return crypto.createHash('sha256').update(text).digest('hex');
}

function evaluate(html, referenceText, comparisonText) {
  const matches = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi)];
  if (matches.length !== 1) throw new Error('Expected one script in the supplied calculator.');
  const elements = new Map();
  const alerts = [];
  function element(id) {
    if (!elements.has(id)) {
      const context = {
        createImageData(width, height) {
          if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0)
            throw new Error('Invalid image dimensions in the calculator input.');
          return { width, height, data: new Uint8ClampedArray(width * height * 4) };
        },
        putImageData(image) { this.imageData = image; },
      };
      elements.set(id, {
        innerText: '', context, classList: { add() {}, remove() {} },
        addEventListener() {}, getContext() { return context; },
      });
    }
    return elements.get(id);
  }
  class FileReader {
    readAsArrayBuffer(file) {
      this.onload({ target: { result: new TextEncoder().encode(file.text).buffer } });
    }
  }
  const sandbox = {
    document: { getElementById: element }, FileReader, TextDecoder,
    alert: message => alerts.push(message),
    referenceFile: { text: referenceText }, comparisonFile: { text: comparisonText },
  };
  const context = vm.createContext(sandbox);
  vm.runInContext(matches[0][1], context, { timeout: 5000 });
  vm.runInContext('parsePPM(referenceFile, true); parsePPM(comparisonFile, false);', context, { timeout: 5000 });
  if (alerts.length) throw new Error(alerts.join('; '));
  const result = element('mse-result').innerText;
  const reference = element('canvas-ref').context.imageData;
  const comparison = element('canvas-upload').context.imageData;
  if (!reference || !comparison || !/^MSE: /.test(result)) throw new Error('Calculator did not return MSE.');
  const sameSize = reference.width === comparison.width && reference.height === comparison.height;
  return {
    displayed_result: result,
    mse_displayed: Number(result.slice(5)),
    reference_size: [reference.width, reference.height],
    comparison_size: [comparison.width, comparison.height],
    identical_dimensions: sameSize,
    valid_full_image_comparison: sameSize,
    compared_pixels: Math.min(reference.width, comparison.width) * Math.min(reference.height, comparison.height),
    size_warning: sameSize ? null : 'Original calculator uses a prefix of flattened arrays for different dimensions. It does not resize or align the images.',
  };
}

function ppm(rgb, width, height) {
  return `P3\n${width} ${height}\n255\n${rgb.flat().join(' ')}\n`;
}

function selfTest(html) {
  assert.equal(evaluate(html, ppm([[0,0,0]],1,1), ppm([[0,0,0]],1,1)).mse_displayed, 0);
  assert.equal(evaluate(html, ppm([[0,0,0]],1,1), ppm([[255,255,255]],1,1)).mse_displayed, 65025);
  // Two pixels: only R differs by three, so (9+9)/(2*3) = 3.
  assert.equal(evaluate(html, ppm([[10,20,30],[40,50,60]],2,1), ppm([[13,20,30],[43,50,60]],2,1)).mse_displayed, 3);
  // Same four values, different row widths: original script returns zero on the prefix.
  const values = [[10,20,30],[40,50,60],[70,80,90],[100,110,120]];
  const mismatch = evaluate(html, ppm(values,2,2), ppm(values,4,1));
  assert.equal(mismatch.mse_displayed, 0);
  assert.equal(mismatch.valid_full_image_comparison, false);
}

if (require.main === module) {
  const [htmlPath, referencePath, comparisonPath] = process.argv.slice(2);
  if (!htmlPath || !referencePath || !comparisonPath) {
    console.error('Usage: node run_teacher_calculator.cjs calculator.html reference.ppm comparison.ppm');
    process.exit(2);
  }
  const [html, reference, comparison] = [htmlPath, referencePath, comparisonPath].map(path => fs.readFileSync(path, 'utf8'));
  selfTest(html);
  console.log(JSON.stringify({
    calculator_sha256: sha256(html), reference_sha256: sha256(reference), comparison_sha256: sha256(comparison),
    engine: 'Original supplied calculator JS executed with canvas ImageData stub; no DOM rendering required for MSE.',
    ...evaluate(html, reference, comparison),
  }, null, 2));
}

module.exports = { evaluate, selfTest, ppm };
