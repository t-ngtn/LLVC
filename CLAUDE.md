# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLVC (Low-Latency Low-Resource Voice Conversion) is a real-time voice conversion system designed for CPU inference with minimal latency (~15ms algorithmic + processing time). The model uses a streaming architecture with causal convolutions and chunked processing to enable real-time conversion.

Paper: https://koe.ai/papers/llvc.pdf

## Setup and Dependencies

### Initial Setup
```bash
# Create environment
conda create -n llvc python=3.11
conda activate llvc

# Install PyTorch (see https://pytorch.org/get-started/locally/)
pip install torch torchaudio

# Install dependencies
pip install -r requirements.txt

# Download pretrained models from HuggingFace
python download_models.py
```

### Evaluation Environment
`eval.py` has conflicting requirements. Create a separate environment:
```bash
conda create -n llvc-eval python=3.9
conda activate llvc-eval
pip install -r eval_requirements.txt
```

## Common Commands

### Inference

**Basic inference** (convert single file or folder):
```bash
python infer.py -p llvc_models/models/checkpoints/llvc/G_500000.pth \
                -c experiments/llvc/config.json \
                -f test_wavs \
                -o converted_out
```

**Streaming inference** (simulates real-time chunked processing):
```bash
python infer.py -p <checkpoint> -c <config> -f <input> -o <output> -s
```

**Streaming with custom chunk size**:
```bash
python infer.py -p <checkpoint> -c <config> -f <input> -o <output> -s -n 2
```
- `-s`: Enable streaming mode
- `-n <chunk_factor>`: Adjust chunk size (trades latency for RTF)
  - Larger values = higher latency, better RTF
  - Default: 1 (chunk_len = 13 * 16 * 1 = 208 samples ≈ 13ms @ 16kHz)

**Compare with RVC/QuickVC streaming**:
```bash
python compare_infer.py
```
Edit `window_ms` and `extra_convert_size` for different streaming parameters.

### Training

1. Create experiment directory with config:
```bash
mkdir -p experiments/my_run
# Copy and modify experiments/llvc/config.json
```

2. Start training:
```bash
python train.py -d experiments/my_run
```

3. Monitor with TensorBoard:
```bash
tensorboard --logdir experiments/my_run/logs
```

Checkpoints saved to: `experiments/my_run/checkpoints/G_<step>.pth`

### Dataset Creation

Datasets require `train/`, `val/`, and `dev/` folders with paired audio:
- `PREFIX_original.wav`: Original speaker audio
- `PREFIX_converted.wav`: Target speaker audio

**Recreate paper dataset**:
```bash
# Download LibriSpeech
# train-clean-360.tar.gz and dev-clean.tar.gz from https://www.openslr.org/12

python -m minimal_rvc._infer_folder \
    --train_set_path "LibriSpeech/train-clean-360" \
    --dev_set_path "LibriSpeech/dev-clean" \
    --out_path "f_8312_ls360" \
    --flatten \
    --model_path "llvc_models/models/rvc/f_8312_32k-325.pth" \
    --model_name "f_8312" \
    --target_sr 16000 \
    --f0_method "rmvpe" \
    --val_percent 0.02 \
    --random_seed 42 \
    --f0_up_key 12
```

### Evaluation

```bash
# Convert test set
python infer.py -p <checkpoint> -c <config> -f test-clean -o test_converted

# Activate eval environment
conda activate llvc-eval

# Run evaluation metrics
python eval.py --converted test_converted --ground_truth test-clean
```

## Architecture Overview

### Model Components

**Main Model (`model.py`)**: The `Net` class implements the streaming voice conversion architecture:

1. **Optional Preprocessing**: `CachedConvNet` (convnet_pre) - Causal conv preprocessing with context buffering
2. **Encoder**: `DilatedCausalConvEncoder` - Dilated causal convs with exponentially increasing dilation (2^i)
3. **Decoder**: `CausalTransformerDecoder` - Processes features in chunks with positional encoding
4. **Mask Generation**: `MaskNet` - Generates time-domain masks for voice conversion

**Key Architectural Features**:
- **Streaming Support**: All layers maintain buffers (`enc_buf`, `dec_buf`, `out_buf`, `convnet_pre_ctx`)
- **Causal Processing**: No future information used (essential for real-time)
- **Chunked Inference**: Decoder processes `dec_chunk_size` chunks at a time
- **Lookahead Context**: Optional 2*L samples from previous chunk

### Streaming Inference Pipeline

The `infer_stream()` function in `infer.py` (lines 37-89) implements chunked processing:

1. **Chunk Creation**: Split audio into `chunk_len = dec_chunk_size * L * chunk_factor` segments
2. **Lookahead Addition**: Add 2*L samples from previous chunk for context
3. **Buffer Initialization**: `model.init_buffers()` creates enc/dec/out buffers
4. **Sequential Processing**: Process each chunk, maintaining buffer state
5. **Metrics Calculation**:
   - RTF (Real-Time Factor): `(chunk_len / sr) / processing_time`
   - End-to-end latency: `((2*L + chunk_len)/sr + processing_time) * 1000` ms

### Configuration Structure

Configs in `experiments/*/config.json` define model architecture and training:

**Critical Model Parameters**:
- `L`: Hop length for convolutions (default: 16)
- `dec_chunk_size`: Decoder chunk size (default: 13)
- `dec_buf_len`: Decoder buffer length (default: 13)
- `num_enc_layers`: Encoder depth (default: 8)
- `num_dec_layers`: Decoder depth (default: 1)
- `lookahead`: Enable lookahead context (default: true)

**Training Parameters**:
- `batch_size`: Training batch size
- `checkpoint_interval`: Steps between checkpoints (default: 5000)
- `log_interval`: Steps between TensorBoard logs (default: 1000)
- `aux_mel`: Multi-resolution mel-spectrogram loss config
- `aux_fairseq`: HuBERT feature matching loss config
- `discriminator`: "rvc" or "hfg" discriminator type

### Related Model Implementations

**minimal_rvc/**: RVC (Retrieval-based Voice Conversion) implementation for:
- Dataset creation via RVC conversion
- Baseline comparisons in `compare_infer.py`
- Key file: `pipeline.py` (VocalConvertPipeline class)

**minimal_quickvc/**: QuickVC baseline implementation for comparisons

Both use chunked inference with SOLA (Similarity Overlap-Add) for crossfading.

## Important Implementation Details

### Sample Rate

LLVC is **fixed at 16kHz** sample rate. All audio is automatically resampled:
```python
audio = torchaudio.transforms.Resample(sr, 16000)(audio)
```

### Streaming vs Non-Streaming Inference

**Non-streaming** (`infer()`):
- Processes entire audio in one forward pass
- Lower memory, no chunking overhead
- No RTF/latency metrics

**Streaming** (`infer_stream()`):
- Simulates real-time processing with chunks
- Maintains buffer state across chunks
- Reports RTF and end-to-end latency
- **Does not currently support real-time microphone I/O**

### Buffer Management

Models maintain four types of buffers for streaming:
1. **enc_buf**: Encoder context buffer (dilated conv history)
2. **dec_buf**: Decoder context buffer (transformer state)
3. **out_buf**: Output buffer for overlap-add
4. **convnet_pre_ctx**: Optional preprocessing context

Initialize with: `enc_buf, dec_buf, out_buf = model.init_buffers(batch_size, device)`

If model has convnet_pre: `convnet_pre_ctx = model.convnet_pre.init_ctx_buf(batch_size, device)`

### Distributed Training

Training uses PyTorch DDP (DistributedDataParallel):
- Multi-GPU automatically detected: `gpus = [i for i in range(torch.cuda.device_count())]`
- Uses gloo backend (for CPU compatibility)
- Master port: 12355
- Spawns one process per GPU with `torch.multiprocessing.spawn()`

### Checkpoint Format

Checkpoints (`.pth` files) contain:
```python
{
    'model': state_dict,
    'optimizer': optimizer_state_dict,
    'learning_rate': float,
    'epoch': int,
    'step': int
}
```

Load with: `model.load_state_dict(torch.load(path)['model'])`

### Discriminators

Two discriminator options controlled by config `"discriminator"` field:
- **"rvc"**: `MultiPeriodDiscriminator` from `discriminators.py` with configurable periods
- **"hfg"**: `ComboDisc` from `hfg_disc.py` (HiFi-GAN style)

Both implement: `discriminator_loss()`, `generator_loss()`, `feature_loss()`

## Testing and Validation

The training loop automatically validates on `dev` and `val` splits:
- **val**: Same speakers as training (measures overfitting)
- **dev**: Different speakers (measures generalization)

Metrics logged to TensorBoard:
- Mel-spectrogram loss
- Fairseq (HuBERT) feature loss
- MCD (Mel-Cepstral Distortion)
- Discriminator losses
- Audio samples (input, ground truth, prediction)

## File Organization

```
experiments/          # Training configurations and outputs
├── llvc/            # Main LLVC config
├── llvc_hfg/        # LLVC with HiFi-GAN discriminator
└── llvc_nc/         # LLVC variant (no convnet)

llvc_models/         # Downloaded pretrained models
├── models/
│   ├── checkpoints/ # LLVC checkpoints
│   ├── embeddings/  # HuBERT models for training
│   ├── rvc/         # RVC models
│   └── rvc_no_f0/   # RVC without pitch modeling

test_wavs/           # Test audio files for quick inference

minimal_rvc/         # RVC implementation for dataset creation
minimal_quickvc/     # QuickVC implementation for baselines
```

## Known Constraints

1. **Fixed 16kHz Sample Rate**: Cannot change without retraining
2. **Mono Audio Only**: Multi-channel audio converted to mono
3. **CPU-Optimized**: Model designed for CPU inference (CUDA supported but not required)
4. **Single Target Speaker**: Each checkpoint converts to one target voice
5. **No Real-Time I/O**: Current implementation processes files, not live audio streams

## Development Notes

- Model uses causal operations throughout - never modify to non-causal
- When adding layers, ensure they support buffer state for streaming
- Chunk size (`dec_chunk_size`) affects latency/quality tradeoff
- RTF < 1.0 is required for real-time (lower is better)
- Test both streaming and non-streaming inference modes
- Fairseq checkpoint auto-downloads if missing during training
