import ddf.minim.*;
import java.util.Arrays;                // 配列を簡単に操作するために使用
import javax.sound.sampled.AudioFormat; // フォーマット情報の取得に使用

// ============================================================
// Minim Instances
// ============================================================
Minim minim;
AudioPlayer player;
AudioFormat format;

// ============================================================
// Sound file（サウンドファイルへのパスを自分の環境に合わせて書く）
// ============================================================
String soundFile = "StratoS.wav"; // data/StratoS.wav を読み込む

// ============================================================
// Audio playback parameters
// ============================================================
final int BUFFER_SIZE = 1024;
float sampleRate;
float outputRate;

// ============================================================
// Low Pass Filter
// ============================================================
LowPassFilter lowPass;
float cutoffFreq; // カットオフ周波数

// ============================================================
// Down Sampling
// ============================================================
final int DOWNSAMPLE_FACTOR = 4; // ここを変えてみる
float downsampledSample = 0.0;

// ============================================================
// Bit Crusher
// ============================================================
BitCrusher bitCrusher;
final int BIT_DEPTH = 8; // 量子化ビット数（ここを変えてみる）

// ============================================================
// 波形表示用リングバッファ
// ============================================================
final int DISPLAY_SAMPLES = 1000;

// Original
float[] originalWave = new float[DISPLAY_SAMPLES];
// LP後
float[] filteredWave = new float[DISPLAY_SAMPLES];
// LP + DS後
float[] downsampledWave = new float[DISPLAY_SAMPLES];
// LP + DS + BC後
float[] bitCrushedWave = new float[DISPLAY_SAMPLES];

int waveIndex = 0;
int waveCount = 0;

// ============================================================
// 表示モード
// 0 : Original
// 1 : Low Pass
// 2 : Low Pass + Down Sampling
// 3 : Low Pass + Down Sampling + Bit Crush
// ============================================================
int displayMode = 0;

// ============================================================
// setup
// ============================================================
void setup() {
  pixelDensity(2); // for Apple Retina Display
  size(1000, 700);

  minim = new Minim(this); // Minim のインスタンスを生成

  println("Space : Pause / Resume");
  println("R : Replay");
  println("O : Original → LP → LP+DS → LP+DS+BC");

  playFile();  // ファイル再生関数
}

// ============================================================
// ファイル再生関数
// ============================================================
void playFile() {
  // ----------------------------------------------------------
  // 現在のファイルを終了
  // ----------------------------------------------------------
  if (player != null) {
    player.close();
    player = null;
  }

  // ----------------------------------------------------------
  // 波形バッファを0で初期化
  // ----------------------------------------------------------
  synchronized (this) { // この処理をしているときに他の処理が配列にアクセスできなくする
    Arrays.fill(originalWave, 0);
    Arrays.fill(filteredWave, 0);
    Arrays.fill(downsampledWave, 0);
    Arrays.fill(bitCrushedWave, 0);

    waveIndex = 0;
    waveCount = 0;
  }

  // ----------------------------------------------------------
  // DS状態
  // ----------------------------------------------------------
  downsampledSample = 0.0;
  downSampler.reset();

  // ----------------------------------------------------------
  // BC
  // ----------------------------------------------------------
  bitCrusher = new BitCrusher(BIT_DEPTH);

  // ----------------------------------------------------------
  // ファイル読み込み
  // ----------------------------------------------------------
  player = minim.loadFile(soundFile, BUFFER_SIZE);

  if (player == null) {
    println("Could not load " + soundFile);
    return;
  }

  // ----------------------------------------------------------
  // AudioFormat インスタンスへ情報取得
  // ----------------------------------------------------------
  format = player.getFormat();
  sampleRate = format.getSampleRate();      // サンプリングレート (Hz)
  outputRate = sampleRate / DOWNSAMPLE_FACTOR;

  // ----------------------------------------------------------
  // LPF
  // ----------------------------------------------------------
  // Down Sampling後のナイキスト周波数より少し低い位置にカットオフを設定
  cutoffFreq = 0.8 * outputRate / 2.0;
  lowPass = new LowPassFilter(cutoffFreq, sampleRate);

  // ----------------------------------------------------------
  // AudioListener
  // ----------------------------------------------------------
  player.addListener(new MyAudioListener());

  println("--- Audio Format Info ---");
  println("The input's format: " + format);
  println("Playing : " + soundFile);
  println("Sample rate : " + sampleRate + " Hz");
  println("Downsample factor : " + DOWNSAMPLE_FACTOR);
  println("Output sample rate : " + (sampleRate / DOWNSAMPLE_FACTOR) + " Hz");
  println("Bit depth : " + BIT_DEPTH + " bit");

  // ----------------------------------------------------------
  // 再生開始
  // ----------------------------------------------------------
  player.play();
}


// ============================================================
// AudioListener
// ============================================================
class MyAudioListener
  implements AudioListener {

  // ----------------------------------------------------------
  // Stereo
  // ----------------------------------------------------------
  public void samples(float[] sampL, float[] sampR) {
    for (int i = 0; i < sampL.length; i++) {
      // Stereo → Mono
      float mono = 0.5 * (sampL[i] + sampR[i]);
      processSample(mono);
    }
  }

  // ----------------------------------------------------------
  // Mono
  // ----------------------------------------------------------
  public void samples(float[] samp) {
    for (int i = 0; i < samp.length; i++) {
      processSample(samp[i]);
    }
  }
}

// ============================================================
// 1 sampleごとの処理
// ============================================================
void processSample(float sample) {

  // ----------------------------------------------------------
  // LP
  // ----------------------------------------------------------
  float filtered = lowPass.process(sample);

  // ----------------------------------------------------------
  // Down Sampling
  // ----------------------------------------------------------
  boolean newSample = downSampler.process(filtered);

  // ----------------------------------------------------------
  // DS出力を保持
  // ----------------------------------------------------------
  if (newSample) {
    downsampledSample = downSampler.getSample();
  }

  // ----------------------------------------------------------
  // Bit Crush
  // ----------------------------------------------------------
  float bitCrushed = bitCrusher.process(downsampledSample);

  // ----------------------------------------------------------
  // 波形保存
  // ----------------------------------------------------------
  synchronized (this) {

    // Original
    originalWave[waveIndex] = sample;
    // LP
    filteredWave[waveIndex] = filtered;
    // LP + DS
    downsampledWave[waveIndex] = downsampledSample;
    // LP + DS + BC
    bitCrushedWave[waveIndex] = bitCrushed;

    // 次の位置へ
    waveIndex++;

    if (waveIndex >= DISPLAY_SAMPLES) {
      waveIndex = 0;
    }
    if (waveCount < DISPLAY_SAMPLES) {
      waveCount++;
    }
  }
}

// ============================================================
// DownSampler
// ============================================================
DownSampler downSampler = new DownSampler(DOWNSAMPLE_FACTOR);

class DownSampler {
  int factor;
  int counter = 0;
  float lastSample = 0;

  DownSampler(int downsampleFactor) {
    factor = downsampleFactor;
  }

  boolean process(float input) {

    counter++;

    if (counter >= factor) {
      counter = 0;
      lastSample = input;
      return true;
    }
    return false;
  }

  float getSample() {
    return lastSample;
  }

  void reset() {
    counter = 0;
    lastSample = 0;
  }
}

// ============================================================
// Low Pass Filter
// ============================================================
class LowPassFilter {
  float alpha;
  float previous = 0.0;

  LowPassFilter(float cutoff, float sampleRate) {
    float ts = 1.0 / sampleRate;
    float tau = 1.0 / (2.0 * PI * cutoff);
    alpha = ts / (tau + ts);
  }

  float process(float input) {
    previous = previous + alpha * (input - previous);
    return previous;
  }

  void reset() {
    previous = 0;
  }
}

// ============================================================
// Bit Crusher
// ============================================================
class BitCrusher {
  int bits;
  float step;

  BitCrusher(int bitDepth) {
    bits = bitDepth;

    // -1 ～ +1 の範囲を 2^bits 段階に量子化する場合のステップ幅
    step = 2.0 / pow(2, bits);
  }

  float process(float input) {
    // 入力範囲を -1 ～ +1 に制限
    input = constrain(input, -1.0, 1.0);

    // 指定ビット数で量子化
    float quantized = round(input / step) * step;

    // 念のため範囲を制限
    quantized = constrain(quantized, -1.0, 1.0);
    return quantized;
  }
}

// ============================================================
// 波形描画
// ============================================================
void drawWaveform() {
  float top = 430;
  float bottom = 650;

  // ----------------------------------------------------------
  // 中心線
  // ----------------------------------------------------------

  stroke(180);
  line(0, (top + bottom) / 2, width, (top + bottom) / 2);

  if (waveCount == 0) {
    return;
  }

  stroke(0);
  noFill();

  beginShape();

  synchronized (this) {
    for (int i = 0; i < waveCount; i++) {
      int index;

      // まだバッファが一杯でない
      if (waveCount < DISPLAY_SAMPLES) {
        index = i;
      } else {
        // 最も古いサンプルから描画
        index = (waveIndex + i) % DISPLAY_SAMPLES;
      }

      float value;

      if (displayMode == 0) {
        value = originalWave[index];
      } else if (displayMode == 1) {
        value = filteredWave[index];
      } else if (displayMode == 2) {
        value = downsampledWave[index];
      } else {
        value = bitCrushedWave[index];
      }

      float x = map(i, 0, max(1, waveCount - 1), 0, width);
      float y = map(value, -1.0, 1.0, bottom, top);

      vertex(x, y);
    }
  }
  endShape();
}

// ============================================================
// draw
// ============================================================
void draw() {
  background(255);
  fill(0);
  textSize(18);

  // ----------------------------------------------------------
  // ファイル
  // ----------------------------------------------------------
  if (player != null) {
    text("File : " + soundFile, 20, 30);
    text("Playing : " + player.isPlaying(), 20, 60);
    text("Position : " + player.position() + " / "
      + player.length() + " ms", 20, 90);
  } else {
    text("No file selected.", 20, 30);
  }


  // ----------------------------------------------------------
  // Sampling
  // ----------------------------------------------------------
  if (player != null && format != null) {
    text("Input file format : " + format, 20, 125);
    text("Downsample factor : " + DOWNSAMPLE_FACTOR, 20, 155);
    text("Output sample rate : " + (sampleRate / DOWNSAMPLE_FACTOR)
      + " Hz", 20, 180);
  }

  // ----------------------------------------------------------
  // LPF / BC
  // ----------------------------------------------------------
  text("LPF cutoff : " + cutoffFreq + " Hz", 20, 215);
  text("Bit depth : " + BIT_DEPTH + " bit", 20, 245);

  // ----------------------------------------------------------
  // Display mode
  // ----------------------------------------------------------
  String modeName;

  if (displayMode == 0) {
    modeName = "ORIGINAL";
  } else if (displayMode == 1) {
    modeName = "LOW PASS";
  } else if (displayMode == 2) {
    modeName = "LOW PASS + DOWN SAMPLE";
  } else {
    modeName = "LOW PASS + DOWN SAMPLE + BIT CRUSH";
  }
  fill(0, 0, 255);
  text("Display : " + modeName, 20, 280);

  // ----------------------------------------------------------
  // 操作
  // ----------------------------------------------------------
  fill(0);
  text("Space : Pause / Resume", 20, 320);
  text("R : Replay", 20, 350);
  text("O : Original → LP → LP+DS → LP+DS+BC", 20, 380);

  // ----------------------------------------------------------
  // 波形
  // ----------------------------------------------------------
  drawWaveform();
}

// ============================================================
// Keyboard
// ============================================================
void keyPressed() {
  // ----------------------------------------------------------
  // Space : Pause / Resume
  // ----------------------------------------------------------

  if (key == ' ') {
    if (player != null) {
      if (player.isPlaying()) {
        player.pause();
      } else {
        if (player.position() < player.length()) {
          player.play();
        }
      }
    }
    return;
  }

  // ----------------------------------------------------------
  // R : Replay
  // ----------------------------------------------------------

  if (key == 'r' || key == 'R') {

    if (player != null) {
      player.pause();
      player.rewind();

      // LPF状態をリセット
      if (lowPass != null) {
        lowPass.reset();
      }

      // DS状態をリセット
      downSampler.reset();
      downsampledSample = 0;

      // 波形をクリア
      synchronized (this) {
        Arrays.fill(originalWave, 0);
        Arrays.fill(filteredWave, 0);
        Arrays.fill(downsampledWave, 0);
        Arrays.fill(bitCrushedWave, 0);

        waveIndex = 0;
        waveCount = 0;
      }
      player.play();
    }
    return;
  }

  // ----------------------------------------------------------
  // O : 表示モード切り替え
  // ----------------------------------------------------------
  if (key == 'o' || key == 'O') {
    displayMode++;

    if (displayMode >= 4) {
      displayMode = 0;
    }
    println("Display mode = " + displayMode);
    return;
  }
}

// ============================================================
// stop
// ============================================================
void stop() {
  if (player != null) {
    player.close();
  }
  if (minim != null) {
    minim.stop();
  }
  super.stop();
}
