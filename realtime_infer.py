"""
Real-time voice conversion using LLVC with microphone input.

This module implements real-time audio processing with:
- Microphone input via sounddevice
- Chunked LLVC inference
- Audio output to speakers
"""

import sounddevice as sd
import torch
import numpy as np
from collections import deque
import threading
import queue
import time
import argparse
import json
from model import Net
from infer import load_model, infer_stream_chunk


class RealtimeVoiceConverter:
    """Real-time voice conversion engine using LLVC model."""

    def __init__(self, model, sample_rate=16000, block_size=512, chunk_factor=1, device='cpu'):
        """
        Initialize real-time voice converter.

        Args:
            model: LLVC model
            sample_rate: Audio sample rate (must be 16000 for LLVC)
            block_size: sounddevice block size in samples
            chunk_factor: LLVC chunk size multiplier
            device: torch device ('cpu', 'cuda', 'mps')
        """
        self.model = model.to(device).eval()
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.chunk_factor = chunk_factor
        self.device = device

        # Calculate chunk sizes
        self.L = model.L
        self.chunk_len = model.dec_chunk_size * self.L * chunk_factor

        # Thread-safe buffers
        self.input_buffer = deque(maxlen=self.chunk_len * 10)
        self.output_queue = queue.Queue(maxsize=20)

        # Model buffers (initialized on start)
        self.enc_buf = None
        self.dec_buf = None
        self.out_buf = None
        self.convnet_pre_ctx = None

        # Lookahead context buffer (previous chunk for context)
        self.prev_chunk = None

        # Statistics
        self.rtf_history = deque(maxlen=100)
        self.underrun_count = 0
        self.chunk_count = 0
        self.running = False

        # Thread
        self.proc_thread = None

    def audio_callback(self, indata, outdata, frames, time_info, status):
        """
        Combined audio I/O callback (runs in high-priority audio thread).

        Args:
            indata: Input audio samples from microphone
            outdata: Output buffer for speaker
            frames: Number of frames
            time_info: Timing information
            status: Status flags
        """
        if status:
            print(f"Audio callback status: {status}")

        # Input: Add microphone samples to buffer
        if indata is not None:
            audio_mono = indata[:, 0].copy()
            self.input_buffer.extend(audio_mono)

        # Output: Retrieve converted audio from queue
        try:
            chunk = self.output_queue.get_nowait()
            chunk_len = min(len(chunk), frames)
            outdata[:chunk_len, 0] = chunk[:chunk_len]
            if chunk_len < frames:
                outdata[chunk_len:, 0] = 0  # Pad with silence
        except queue.Empty:
            # Underrun: no audio available
            outdata[:, 0] = 0
            self.underrun_count += 1

    def processing_thread_func(self):
        """
        Audio processing thread (runs in separate thread).

        Continuously reads from input buffer, processes through LLVC,
        and writes to output queue.
        """
        while self.running:
            # Wait for sufficient samples
            if len(self.input_buffer) < self.chunk_len:
                time.sleep(0.001)  # 1ms sleep
                continue

            # Extract chunk from input buffer
            chunk = np.array([self.input_buffer.popleft()
                             for _ in range(self.chunk_len)])

            # Add lookahead context
            if self.prev_chunk is not None:
                lookahead_ctx = self.prev_chunk[-self.L * 2:]
            else:
                lookahead_ctx = np.zeros(self.L * 2, dtype=np.float32)

            chunk_with_ctx = np.concatenate([lookahead_ctx, chunk])
            self.prev_chunk = chunk.copy()

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(chunk_with_ctx).float().to(self.device)

            # Process through LLVC model
            start_time = time.time()
            output, self.enc_buf, self.dec_buf, self.out_buf, self.convnet_pre_ctx = \
                infer_stream_chunk(
                    self.model,
                    audio_tensor,
                    self.enc_buf,
                    self.dec_buf,
                    self.out_buf,
                    self.convnet_pre_ctx
                )
            processing_time = time.time() - start_time

            # Calculate RTF
            rtf = processing_time / (self.chunk_len / self.sample_rate)
            self.rtf_history.append(rtf)
            self.chunk_count += 1

            # Extract output audio (remove batch and channel dimensions)
            output_audio = output.squeeze(0).squeeze(0).cpu().numpy()

            # Add to output queue
            try:
                self.output_queue.put(output_audio, timeout=0.1)
            except queue.Full:
                print("Warning: Output queue full, dropping chunk")

    def start(self, input_device=None, output_device=None):
        """
        Start real-time voice conversion.

        Args:
            input_device: Input device ID (None for default)
            output_device: Output device ID (None for default)
        """
        print("=" * 60)
        print("Real-time Voice Conversion")
        print("=" * 60)
        print(f"Model: LLVC")
        print(f"Sample Rate: {self.sample_rate} Hz")
        print(f"Block Size: {self.block_size} samples ({self.block_size/self.sample_rate*1000:.1f} ms)")
        print(f"Chunk Size: {self.chunk_len} samples ({self.chunk_len/self.sample_rate*1000:.1f} ms)")
        print(f"Algorithmic Latency: ~{(2*self.L + self.chunk_len)/self.sample_rate*1000:.1f} ms")
        print(f"Device: {self.device}")
        print()

        if input_device is not None:
            print(f"Input Device: {sd.query_devices(input_device)['name']}")
        else:
            print(f"Input Device: Default")

        if output_device is not None:
            print(f"Output Device: {sd.query_devices(output_device)['name']}")
        else:
            print(f"Output Device: Default")
        print()

        self.running = True

        # Initialize model buffers
        device_obj = torch.device(self.device)
        self.enc_buf, self.dec_buf, self.out_buf = self.model.init_buffers(1, device_obj)
        if hasattr(self.model, 'convnet_pre'):
            self.convnet_pre_ctx = self.model.convnet_pre.init_ctx_buf(1, device_obj)
        else:
            self.convnet_pre_ctx = None

        # Pre-warm model with dummy data
        print("Pre-warming model...")
        dummy_chunk = torch.zeros(self.chunk_len + 2 * self.L).float().to(self.device)
        for _ in range(3):
            _, self.enc_buf, self.dec_buf, self.out_buf, self.convnet_pre_ctx = \
                infer_stream_chunk(
                    self.model, dummy_chunk,
                    self.enc_buf, self.dec_buf, self.out_buf, self.convnet_pre_ctx
                )
        print("Pre-warming complete.")
        print()

        # Start processing thread
        self.proc_thread = threading.Thread(target=self.processing_thread_func, daemon=True)
        self.proc_thread.start()

        # Start audio stream
        print("Starting conversion... Press Ctrl+C to stop.")
        print()

        try:
            with sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=(input_device, output_device),
                channels=1,
                dtype='float32',
                callback=self.audio_callback
            ):
                # Main loop: display statistics
                last_print_time = time.time()
                while self.running:
                    current_time = time.time()
                    if current_time - last_print_time >= 1.0:  # Update every second
                        if len(self.rtf_history) > 0:
                            avg_rtf = np.mean(self.rtf_history)
                            max_rtf = np.max(self.rtf_history)
                            input_buf_len = len(self.input_buffer)
                            output_queue_len = self.output_queue.qsize()

                            print(f"\rRTF: {avg_rtf:.3f} (max: {max_rtf:.3f}) | "
                                  f"Chunks: {self.chunk_count} | "
                                  f"Underruns: {self.underrun_count} | "
                                  f"InBuf: {input_buf_len}/{self.input_buffer.maxlen} | "
                                  f"OutQ: {output_queue_len}/{self.output_queue.maxsize}",
                                  end='', flush=True)
                        last_print_time = current_time
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nStopping...")
            self.stop()

    def stop(self):
        """Stop real-time voice conversion."""
        self.running = False
        if self.proc_thread is not None:
            self.proc_thread.join(timeout=2.0)

        print("\nConversion stopped.")
        print(f"Total chunks processed: {self.chunk_count}")
        print(f"Total underruns: {self.underrun_count}")
        if len(self.rtf_history) > 0:
            print(f"Average RTF: {np.mean(self.rtf_history):.3f}")
            print(f"Max RTF: {np.max(self.rtf_history):.3f}")


def list_audio_devices():
    """List all available audio devices."""
    print("\nAvailable Audio Devices:")
    print("=" * 80)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        device_type = []
        if device['max_input_channels'] > 0:
            device_type.append("Input")
        if device['max_output_channels'] > 0:
            device_type.append("Output")
        print(f"[{i}] {device['name']}")
        print(f"    Type: {', '.join(device_type)}")
        print(f"    Channels: In={device['max_input_channels']}, Out={device['max_output_channels']}")
        print(f"    Sample Rate: {device['default_samplerate']} Hz")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time voice conversion using LLVC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available audio devices
  python realtime_infer.py --list_devices

  # Use default devices
  python realtime_infer.py -p model.pth -c config.json

  # Specify devices
  python realtime_infer.py -p model.pth -c config.json --input_device 0 --output_device 1

  # Adjust chunk size for lower latency
  python realtime_infer.py -p model.pth -c config.json -n 1

  # Use GPU for faster processing
  python realtime_infer.py -p model.pth -c config.json --device cuda
        """
    )
    parser.add_argument(
        '--checkpoint_path', '-p', type=str,
        default='llvc_models/models/checkpoints/llvc/G_500000.pth',
        help='Path to LLVC checkpoint file'
    )
    parser.add_argument(
        '--config_path', '-c', type=str,
        default='experiments/llvc/config.json',
        help='Path to LLVC config file'
    )
    parser.add_argument(
        '--input_device', '-i', type=int, default=None,
        help='Input device ID (use --list_devices to see options)'
    )
    parser.add_argument(
        '--output_device', '-o', type=int, default=None,
        help='Output device ID (use --list_devices to see options)'
    )
    parser.add_argument(
        '--block_size', '-b', type=int, default=512,
        help='sounddevice block size in samples (default: 512 ≈ 32ms @ 16kHz)'
    )
    parser.add_argument(
        '--chunk_factor', '-n', type=int, default=1,
        help='LLVC chunk size multiplier (default: 1)'
    )
    parser.add_argument(
        '--device', '-d', type=str, default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Torch device for inference (default: cpu)'
    )
    parser.add_argument(
        '--list_devices', action='store_true',
        help='List available audio devices and exit'
    )

    args = parser.parse_args()

    # List devices and exit
    if args.list_devices:
        list_audio_devices()
        return

    # Load model
    print("Loading model...")
    model, sr = load_model(args.checkpoint_path, args.config_path)

    if sr != 16000:
        print(f"Warning: Model sample rate is {sr}, but LLVC requires 16000 Hz")
        print("Using 16000 Hz for real-time inference")
        sr = 16000

    print(f"Model loaded successfully from {args.checkpoint_path}")
    print()

    # Create converter
    converter = RealtimeVoiceConverter(
        model=model,
        sample_rate=sr,
        block_size=args.block_size,
        chunk_factor=args.chunk_factor,
        device=args.device
    )

    # Start conversion
    converter.start(
        input_device=args.input_device,
        output_device=args.output_device
    )


if __name__ == '__main__':
    main()
