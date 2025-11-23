# Real-time Voice Conversion

リアルタイムでマイク入力を変換してスピーカーから出力する機能が実装されました。

## セットアップ

```bash
# sounddeviceをインストール（既にrequirements.txtに含まれています）
pip install sounddevice==0.4.6
```

## 使い方

### 1. 利用可能なオーディオデバイスを確認

```bash
python realtime_infer.py --list_devices
```

### 2. リアルタイム変換を開始

#### デフォルトデバイスを使用（最も簡単）
```bash
python realtime_infer.py -n 2 --device mps
```

#### 特定のデバイスを指定
```bash
python realtime_infer.py -n 2 --device mps --input_device 2 --output_device 3
```

#### カスタムモデルを使用
```bash
python realtime_infer.py \
    -p llvc_models/models/checkpoints/llvc/G_500000.pth \
    -c experiments/llvc/config.json \
    -n 2 \
    --device mps \
    --input_device 0 \
    --output_device 1
```

### 3. 停止

Ctrl+C を押すことで停止できます。

## パラメータ説明

- `-p, --checkpoint_path`: モデルチェックポイントのパス（デフォルト: `llvc_models/models/checkpoints/llvc/G_500000.pth`）
- `-c, --config_path`: 設定ファイルのパス（デフォルト: `experiments/llvc/config.json`）
- `-i, --input_device`: 入力デバイスID（デフォルト: システムのデフォルト）
- `-o, --output_device`: 出力デバイスID（デフォルト: システムのデフォルト）
- `-b, --block_size`: sounddeviceのブロックサイズ（デフォルト: 512 ≈ 32ms @ 16kHz）
- `-n, --chunk_factor`: LLVCチャンクサイズの倍率（デフォルト: 1、推奨: 2以上）
- `-d, --device`: 推論デバイス（`cpu`, `cuda`, `mps`）

## パフォーマンス最適化

### デバイス別の推奨設定

#### Apple Silicon (M1/M2/M3)
```bash
python realtime_infer.py -n 2 --device mps
```
- RTF: ~0.62
- レイテンシー: ~44ms
- ✓ リアルタイム処理可能

#### CPU
```bash
python realtime_infer.py -n 8 --device cpu
```
- RTF: ~1.3 @ chunk_factor=1（リアルタイム不可）
- chunk_factorを大きくすることで改善可能だが、レイテンシーが増加

#### NVIDIA GPU (CUDA)
```bash
python realtime_infer.py -n 2 --device cuda
```
- 最高のパフォーマンスが期待できます

### Chunk Factor の選択

| chunk_factor | Chunk長 | RTF (MPS) | レイテンシー | リアルタイム |
|--------------|---------|-----------|--------------|--------------|
| 1            | 208 (13ms) | 1.12 | ~30ms | ✗ |
| 2            | 416 (26ms) | 0.62 | ~44ms | ✓ |
| 3            | 624 (39ms) | 0.42 | ~57ms | ✓ |
| 4            | 832 (52ms) | 0.32 | ~71ms | ✓ |
| 5            | 1040 (65ms) | 0.26 | ~84ms | ✓ |
| 8            | 1664 (104ms) | 0.16 | ~123ms | ✓ |

**推奨:**
- 低レイテンシー優先: `chunk_factor=2` (44ms)
- 安定性優先: `chunk_factor=3-4` (57-71ms)
- 高スループット: `chunk_factor=5以上` (84ms+)

## トラブルシューティング

### "RTF > 1.0" - 処理が遅い
- `--device mps` または `--device cuda` を使用してGPU処理を有効化
- `-n` パラメータを増やす（例: `-n 3` or `-n 4`）

### オーディオのアンダーラン（途切れ）
- `-n` パラメータを増やしてチャンクサイズを大きくする
- `-b` パラメータを増やしてブロックサイズを大きくする（例: `-b 1024`）

### デバイスが見つからない
- `--list_devices` でデバイスリストを確認
- デバイスIDを正しく指定しているか確認

### エコー・ハウリング
- ヘッドフォンを使用する
- スピーカーの音量を下げる
- 入力デバイスと出力デバイスを物理的に離す

## テストスクリプト

### 基本的な動作確認
```bash
python test_realtime.py
```

### デバイス別パフォーマンステスト
```bash
python test_realtime_mps.py
```

### 最適なchunk_factorを見つける
```bash
python test_chunk_factors.py
```

## アーキテクチャ

```
┌─────────────┐
│ Microphone  │ (sounddevice input)
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Input Buffer       │ (deque, thread-safe)
│  (audio samples)    │
└──────┬──────────────┘
       │
       ▼ (Processing Thread)
┌─────────────────────┐
│  LLVC Model         │ (chunk-by-chunk processing)
│  - Encoder          │ (with buffer state management)
│  - Decoder          │
│  - MaskNet          │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Output Queue       │ (queue.Queue, thread-safe)
│  (converted audio)  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Speaker Output     │ (sounddevice output)
└─────────────────────┘
```

### スレッドモデル

1. **Audio Callback Thread** (高優先度)
   - sounddeviceが管理
   - マイク入力 → Input Buffer
   - Output Queue → スピーカー出力

2. **Processing Thread** (中優先度)
   - Input Bufferからチャンクを取得
   - LLVCモデルで変換
   - Output Queueに追加

## 実装の詳細

- **ファイル:** `realtime_infer.py`
- **主要クラス:** `RealtimeVoiceConverter`
- **依存:** `sounddevice`, `torch`, `numpy`

### 主要メソッド

- `start()`: リアルタイム変換を開始
- `stop()`: 変換を停止
- `audio_callback()`: オーディオI/Oコールバック
- `processing_thread_func()`: 変換処理スレッド

## 今後の改善案

### Phase 2: 安定性向上
- [ ] Adaptive buffer sizing（RTFに応じた動的調整）
- [ ] より詳細なエラーハンドリング
- [ ] レイテンシー監視と警告

### Phase 3: ユーザビリティ向上
- [ ] GUI（オプション）
- [ ] 設定プリセット（low-latency, balanced, high-quality）
- [ ] オーディオ品質メトリクス表示

### Phase 4: 最適化
- [ ] モデル量子化
- [ ] ノイズゲート
- [ ] AGC (Automatic Gain Control)
- [ ] エコーキャンセレーション

## ライセンス

LLVC本体と同じライセンスが適用されます。
