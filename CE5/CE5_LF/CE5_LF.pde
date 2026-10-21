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

// ============================================================
// Low Pass Filter Instance
// ============================================================
LowPassFilter lowPass;
// カットオフ周波数（任意の値に変える）
float cutoffFreq = 4000; // 4 kHz

// ============================================================
// 波形表示用リングバッファ
// ============================================================
final int DISPLAY_SAMPLES = 1000;

float[] originalWave = new float[DISPLAY_SAMPLES];
float[] filteredWave = new float[DISPLAY_SAMPLES];

int waveIndex = 0;
int waveCount = 0;

// ============================================================
// 表示モード
// true  : LP後
// false : Original
// ============================================================
boolean showLowPass = true;

// ============================================================
// setup
// ============================================================
void setup() {
  pixelDensity(2); // for Apple Retina Display
  size(1000, 650);

  minim = new Minim(this); // Minim のインスタンスを生成

  println("Space : Pause / Resume");
  println("R     : Replay from beginning");
  println("O     : Original / Low Pass");

  playFile();  // ファイル再生関数
}

// ============================================================
// ファイル再生関数
// ============================================================
void playFile() {
  // ------------------------------------------
  // 現在のファイルを終了
  // ------------------------------------------
  if (player != null) {
    player.close();
    player = null;
  }

  // ------------------------------------------
  // 波形バッファを0で初期化
  // ------------------------------------------
  Arrays.fill(originalWave, 0);
  Arrays.fill(filteredWave, 0);

  waveIndex = 0;
  waveCount = 0;

  // ------------------------------------------
  // ファイルを読み込む
  // ------------------------------------------
  player = minim.loadFile(soundFile, BUFFER_SIZE);

  if (player == null) {
    println("Could not load " + soundFile);
    return;
  }

  // ------------------------------------------
  //  AudioFormat インスタンスへ情報取得
  // ------------------------------------------
  format = player.getFormat();
  sampleRate = format.getSampleRate(); // サンプリングレート (Hz)

  // ------------------------------------------
  // Low Pass Filter インスタンスを生成
  // ------------------------------------------
  lowPass = new LowPassFilter(cutoffFreq, sampleRate);

  // ------------------------------------------
  // AudioListenerを登録
  // ------------------------------------------
  player.addListener(new MyAudioListener());

  println("--- Audio Format Info ---");
  println("The input's format: " + format);
  println("Playing : " + soundFile);
  println("Sample rate : " + sampleRate);

  // ------------------------------------------
  // 再生
  // ------------------------------------------
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

  // ------------------------------------------
  // Low Pass Filter
  // ------------------------------------------
  float filtered = lowPass.process(sample);

  // ------------------------------------------
  // リングバッファへ保存
  // ------------------------------------------
  synchronized (this) {
    originalWave[waveIndex] = sample;
    filteredWave[waveIndex] = filtered;

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
// draw
// ============================================================
void draw() {
  background(255);
  fill(0);
  textSize(18);

  // ----------------------------------------------------------
  // サウンドファイル情報
  // ----------------------------------------------------------
  if (player != null) {
    text("File : " + soundFile, 20, 30);
    text("Playing : " + player.isPlaying(), 20, 60);
    text("Position : " + player.position() + " / "
     + player.length() + " ms", 20, 90);
  } else {
    text("No file selected.", 20, 30);
  }
  text("Input file format : " + format, 20, 125);

  // ----------------------------------------------------------
  // Filter関連
  // ----------------------------------------------------------
  text("Sampling rate : " + sampleRate + " Hz", 20, 155);
  text( "LPF cutoff : " + cutoffFreq + " Hz", 20, 180);
  fill(0, 0, 255);
  text("Display : " + (showLowPass ? "LOW PASS" : "ORIGINAL"), 20, 215);

  // ----------------------------------------------------------
  // 操作関連
  // ----------------------------------------------------------
  fill(0);
  text("Space : Pause / Resume", 20, 260);
  text("R : Replay", 20, 290);
  text("O : Original / Low Pass", 20, 320);

  // ----------------------------------------------------------
  // 波形描画
  // ----------------------------------------------------------
  drawWaveform();
}

// ============================================================
// 波形描画関数
// ============================================================
void drawWaveform() {
  float top = 380;
  float bottom = 620;

  // 中心線
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

      // まだリングバッファが一杯でない
      if (waveCount < DISPLAY_SAMPLES) {
        index = i;
      } else {
        // 最も古いデータから描画
        index = (waveIndex + i) % DISPLAY_SAMPLES;
      }

      float value;

      if (showLowPass) {
        value = filteredWave[index];
      } else {
        value = originalWave[index];
      }

      float x = map(i, 0, max(1, waveCount - 1), 0, width);
      float y = map(value, -1.0, 1.0, bottom, top);

      vertex(x, y);
    }
  }

  endShape();
}

// ============================================================
// キーボード操作
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

      if (lowPass != null) {
        lowPass.reset();
      }

      synchronized (this) {
        Arrays.fill( originalWave, 0);
        Arrays.fill(filteredWave, 0);

        waveIndex = 0;
        waveCount = 0;
      }

      player.play();
    }
    return;
  }

  // ----------------------------------------------------------
  // O : Original / Low Pass
  // ----------------------------------------------------------
  if (key == 'o' || key == 'O') {
    showLowPass = !showLowPass;
    println(showLowPass ? "Display = Low Pass" : "Display = Original");
    return;
  }
}

// ============================================================
// LowPassFilter
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

  void reset() {        // "R" キーが押されたとき
    previous = 0.0;
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
