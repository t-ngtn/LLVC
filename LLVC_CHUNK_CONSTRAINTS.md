# LLVC チャンク処理とリアルタイム制約

LLVCのストリーミング推論における技術的制約と設計の詳細ドキュメント。

## 1. 基本パラメータ

| パラメータ | デフォルト値 | 設定ファイル | 説明 |
|-----------|-------------|--------------|------|
| `L` (hop length) | 16 | `model_params.L` | 畳み込みのホップ長（サンプル数） |
| `dec_chunk_size` | 13 | `model_params.dec_chunk_size` | デコーダのチャンクサイズ |
| `dec_buf_len` | 13 | `model_params.dec_buf_len` | デコーダバッファ長 |
| `out_buf_len` | 4 | `model_params.out_buf_len` | 出力バッファ長 |
| `num_enc_layers` | 8 | `model_params.num_enc_layers` | エンコーダ層数 |
| `sr` | 16000 Hz | `data.sr` | サンプルレート（**固定**） |
| `chunk_factor` | 1 | CLI引数 `-n` | チャンク倍率（実行時調整可能） |

## 2. チャンクサイズの計算

### 計算式

```
chunk_len = dec_chunk_size × L × chunk_factor
```

### デフォルト値での計算

```
chunk_len = 13 × 16 × 1 = 208 サンプル
時間換算 = 208 / 16000 = 0.013秒 = 13ms
```

### chunk_factor別のチャンク長

| chunk_factor | chunk_len (samples) | chunk_len (ms) |
|--------------|---------------------|----------------|
| 1 | 208 | 13 |
| 2 | 416 | 26 |
| 3 | 624 | 39 |
| 4 | 832 | 52 |
| 5 | 1040 | 65 |
| 8 | 1664 | 104 |

## 3. Lookaheadコンテキスト

### 概要

LLVCはlookaheadオプションを使用して、各チャンクの処理時に前のチャンクの末尾データを参照します。

### 実装詳細 (`infer.py:73-81`)

```python
# 各チャンクの先頭に前チャンクの末尾 2*L サンプルを付加
for i, a in enumerate(audio_chunks):
    if i == 0:
        front_ctx = torch.zeros(L * 2)  # 最初のチャンクはゼロパディング
    else:
        front_ctx = audio_chunks[i - 1][-L * 2:]  # 前チャンクの末尾32サンプル
    new_audio_chunks.append(torch.cat([front_ctx, a]))
```

### Lookaheadのサイズ

```
lookahead_samples = 2 × L = 2 × 16 = 32 サンプル = 2ms
```

### 入力チャンクの構造

```
[lookahead context: 2*L] + [current chunk: chunk_len]
= 32 + 208 = 240 サンプル（入力）
→ 208 サンプル（出力）
```

## 4. レイテンシ計算

### エンドツーエンドレイテンシの計算式 (`infer.py:104`)

```
e2e_latency = ((2×L + chunk_len) / sr + processing_time) × 1000 [ms]
```

### 内訳

| 成分 | 計算 | デフォルト値 |
|------|------|-------------|
| Lookaheadコンテキスト | `2×L / sr` | 32/16000 = 2ms |
| チャンク長 | `chunk_len / sr` | 208/16000 = 13ms |
| **アルゴリズム的レイテンシ** | `(2×L + chunk_len) / sr` | **15ms** |
| 処理時間 | 実測値（デバイス依存） | 変動 |

### 理論的最小レイテンシ

処理時間が0と仮定した場合:

| chunk_factor | アルゴリズム的レイテンシ |
|--------------|------------------------|
| 1 | 15ms |
| 2 | 28ms |
| 3 | 41ms |
| 4 | 54ms |

## 5. RTF (Real-Time Factor)

### 定義

```
RTF = (chunk_len / sr) / processing_time
    = チャンク音声長 / 処理時間
```

### 解釈

| RTF値 | 意味 |
|-------|------|
| RTF > 1.0 | リアルタイム処理可能（処理が音声より速い） |
| RTF = 1.0 | ギリギリリアルタイム |
| RTF < 1.0 | リアルタイム処理不可（処理が追いつかない） |

### 目安値

| デバイス | chunk_factor=1 | chunk_factor=2 | chunk_factor=4 |
|----------|----------------|----------------|----------------|
| CPU (M1) | ~0.8 | ~1.3 | ~2.0 |
| MPS (M1) | ~1.1 | ~1.6 | ~3.1 |
| CUDA | ~2.0+ | ~3.0+ | ~5.0+ |

## 6. バッファ管理

### バッファの種類

LLVCのストリーミング推論では4種類のバッファを使用します。

#### 1. Encoder Buffer (`enc_buf`)

Dilated Causal Convolutionの履歴を保持。

```python
# サイズ計算
buf_length = (kernel_size - 1) × (2^num_layers - 1)
           = (3 - 1) × (2^8 - 1)
           = 2 × 255 = 510

# 形状: [batch_size, channels, buf_length]
enc_buf = torch.zeros(batch_size, enc_dim, 510)
```

#### 2. Decoder Buffer (`dec_buf`)

Transformerデコーダの過去のコンテキストを保持。

```python
# 形状: [batch_size, num_layers+1, ctx_len, model_dim]
dec_buf = torch.zeros(batch_size, num_dec_layers+1, dec_buf_len, dec_dim)
        = torch.zeros(batch_size, 2, 13, 256)
```

#### 3. Output Buffer (`out_buf`)

ConvTransposeのoverlap-add用バッファ。

```python
# 形状: [batch_size, enc_dim, out_buf_len]
out_buf = torch.zeros(batch_size, enc_dim, out_buf_len)
        = torch.zeros(batch_size, 512, 4)
```

#### 4. ConvNet Pre Context (`convnet_pre_ctx`)

オプションの前処理ConvNet用コンテキストバッファ。

```python
# CachedConvNetのinit_ctx_buf()で初期化
convnet_pre_ctx = model.convnet_pre.init_ctx_buf(batch_size, device)
```

### バッファの初期化

```python
# メインバッファの初期化
enc_buf, dec_buf, out_buf = model.init_buffers(batch_size, device)

# ConvNet前処理バッファの初期化（モデルにconvnet_preがある場合）
if hasattr(model, 'convnet_pre'):
    convnet_pre_ctx = model.convnet_pre.init_ctx_buf(batch_size, device)
else:
    convnet_pre_ctx = None
```

### ストリーミング時のバッファ更新

```python
# チャンク処理ごとにバッファを更新
output, enc_buf, dec_buf, out_buf, convnet_pre_ctx = model(
    audio_chunk.unsqueeze(0).unsqueeze(0),
    enc_buf, dec_buf, out_buf,
    convnet_pre_ctx,
    pad=(not model.lookahead)
)
```

## 7. 因果処理の制約

### 設計原則

LLVCは完全に因果的（causal）に設計されており、未来の情報は一切使用しません。

### 因果性を保証するコンポーネント

| コンポーネント | 因果性の実現方法 |
|---------------|-----------------|
| `DilatedCausalConvEncoder` | 右側パディングなし、履歴バッファで過去のみ参照 |
| `CausalTransformerDecoder` | 最後のトークンのみ出力、過去のコンテキストのみ参照 |
| `in_conv` | `padding=0`で因果畳み込み |
| `out_conv` | ConvTranspose + overlap-addで因果出力 |

### 因果畳み込みの図解

```
非因果畳み込み（通常）:
  入力:  [past] [current] [future]
           ↓       ↓        ↓
  出力:        [output]

因果畳み込み（LLVC）:
  入力:  [past] [current]
           ↓       ↓
  出力:        [output]
         ※ futureは使用しない
```

## 8. 固定制約

### 変更不可の制約

| 制約 | 詳細 | 理由 |
|------|------|------|
| **サンプルレート 16kHz** | 再学習なしに変更不可 | モデルがこのレートで学習済み |
| **モノラル専用** | ステレオは自動的にモノラル化 | `audio.mean(0)` |
| **単一ターゲット話者** | 1チェックポイント = 1声 | ラベル埋め込みが固定 |
| **因果処理のみ** | 未来の情報は使用しない | リアルタイム要件 |

### サンプルレート変換

```python
# 入力音声は自動的に16kHzにリサンプル
audio = torchaudio.transforms.Resample(original_sr, 16000)(audio)
```

## 9. chunk_factorによるトレードオフ

### パフォーマンス影響

```
chunk_factor ↑ → chunk_len ↑ → バッチ効率 ↑ → RTF ↑
chunk_factor ↑ → chunk_len ↑ → アルゴリズム的レイテンシ ↑
```

### 推奨設定

| ユースケース | chunk_factor | 想定レイテンシ | 備考 |
|-------------|--------------|---------------|------|
| 最低レイテンシ | 1 | ~15ms + 処理時間 | 高性能GPU必須 |
| 低レイテンシ | 2 | ~28ms + 処理時間 | MPS/CUDA推奨 |
| バランス | 3-4 | ~41-54ms + 処理時間 | 安定性重視 |
| 高スループット | 5-8 | ~65-104ms + 処理時間 | CPU向け |

## 10. ストリーミング推論の実装パターン

### 基本パターン

```python
# 1. モデル読み込み
model, sr = load_model(checkpoint_path, config_path)

# 2. バッファ初期化
enc_buf, dec_buf, out_buf = model.init_buffers(1, device)
convnet_pre_ctx = model.convnet_pre.init_ctx_buf(1, device) if hasattr(model, 'convnet_pre') else None

# 3. Lookaheadコンテキスト用の前回チャンク末尾を保持
prev_chunk_tail = torch.zeros(L * 2)

# 4. チャンクごとに処理
for audio_chunk in audio_stream:
    # Lookaheadコンテキストを付加
    chunk_with_ctx = torch.cat([prev_chunk_tail, audio_chunk])

    # 推論
    output, enc_buf, dec_buf, out_buf, convnet_pre_ctx = model(
        chunk_with_ctx.unsqueeze(0).unsqueeze(0),
        enc_buf, dec_buf, out_buf,
        convnet_pre_ctx,
        pad=False
    )

    # 次回用に現在のチャンク末尾を保存
    prev_chunk_tail = audio_chunk[-L * 2:]

    # 出力を再生キューに追加
    output_queue.put(output.squeeze())
```

### リアルタイム実装時の注意点

1. **スレッド安全性**: 入力バッファと出力キューはスレッドセーフに
2. **アンダーラン対策**: 出力キューが空の場合の無音挿入
3. **バッファリング**: 初期バッファリングでグリッチ防止
4. **例外処理**: 処理遅延時のスキップロジック

## 11. メモリ使用量

### バッファサイズの概算

```python
# バッチサイズ1、float32の場合
enc_buf_size = 1 × 512 × 510 × 4 bytes ≈ 1.0 MB
dec_buf_size = 1 × 2 × 13 × 256 × 4 bytes ≈ 26 KB
out_buf_size = 1 × 512 × 4 × 4 bytes ≈ 8 KB

total_buffer ≈ 1.04 MB
```

### モデルサイズ

```
LLVC (G_500000.pth): 約 20-30 MB
```

## 12. 関連ファイル

| ファイル | 説明 |
|----------|------|
| `model.py` | モデルアーキテクチャとバッファ管理 |
| `infer.py` | ファイルベースのストリーミング推論 |
| `realtime_infer.py` | リアルタイムマイク入出力 |
| `experiments/llvc/config.json` | デフォルト設定 |
| `cached_convnet.py` | 前処理ConvNetのバッファ実装 |

## 13. 参考資料

- 論文: [LLVC: Low-Latency Low-Resource Voice Conversion](https://koe.ai/papers/llvc.pdf)
- モデルダウンロード: HuggingFaceから`python download_models.py`で取得
