"""
Voxtral Mini 4B Realtime - CPU Transcription Script
Uses HuggingFace Transformers for local, private speech-to-text.
All processing happens locally - no data leaves your machine.

Usage:
    python transcribe_voxtral.py audio_file.mp3
    python transcribe_voxtral.py audio_file.wav
    python transcribe_voxtral.py --mic
    python transcribe_voxtral.py --help
"""

import argparse
import datetime
import os
import sys
import time
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
from transformers import TextIteratorStreamer
from mistral_common.tokens.tokenizers.audio import Audio

from utils import print_ram, save_transcript, record_until_esc_or_timeout, get_loopback_device

MODEL_NAME = "Voxtral"
VOXTRAL_MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"


def load_model_and_processor(
    model_id: str = VOXTRAL_MODEL_ID,
    delay_ms: int = None,
):
    """Load the Voxtral model and processor on CPU."""
    print_ram("Before loading")

    if delay_ms is not None:
        print(f"\n  Setting Voxtral transcription delay to {delay_ms}ms...")
        _set_transcription_delay(model_id, delay_ms)

    print(f"\n[Step 1/3] Loading Voxtral tokenizer from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    print("[Step 1/3] Tokenizer loaded.")
    print_ram("After tokenizer")

    print("\n[Step 2/3] Loading Voxtral model weights into CPU memory...")
    print("  This takes ~1-2 minutes on CPU. Please wait...")
    t0 = time.time()
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    elapsed = time.time() - t0
    print(f"[Step 2/3] Voxtral model weights loaded in {elapsed:.0f}s.")
    print_ram("After model weights")

    print("\n[Step 3/3] Initializing Voxtral on CPU...")
    t0 = time.time()
    _ = model.device
    elapsed = time.time() - t0
    print(f"[Step 3/3] Voxtral ready in {elapsed:.0f}s.")
    print_ram("After initialization")

    weight_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    print(f"\n{'=' * 40}")
    print(f"  Model:      Voxtral Mini 4B Realtime")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    print(f"  Weights RAM: ~{weight_mem:.1f}GB (BF16)")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype: {next(model.parameters()).dtype}")
    print(f"{'=' * 40}")

    return model, processor


def _set_transcription_delay(model_id: str, delay_ms: int):
    """Modify transcription_delay_ms in the local Voxtral model cache."""
    import json
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(repo_id=model_id)
    tekken_path = os.path.join(model_dir, "tekken.json")

    with open(tekken_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["config"]["transcription_delay_ms"] = delay_ms

    with open(tekken_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"  Transcription delay set to {delay_ms}ms in {tekken_path}")


def transcribe_audio(model, processor, audio_path: str, delay_ms: int = None):
    """Transcribe a single audio file."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    print(f"\nLoading audio: {audio_path.name}")
    audio = Audio.from_file(str(audio_path), strict=False)
    audio.resample(processor.feature_extractor.sampling_rate)
    print(
        f"  Duration: {len(audio.audio_array) / processor.feature_extractor.sampling_rate:.1f}s"
    )
    print(f"  Sample rate: {processor.feature_extractor.sampling_rate}Hz")

    print("\nProcessing audio...")
    inputs = processor(audio.audio_array, return_tensors="pt")
    inputs = inputs.to(model.device, dtype=model.dtype)

    audio_dur = len(audio.audio_array) / processor.feature_extractor.sampling_rate
    expected_tokens = int(audio_dur * 12.5)
    print(f"  Generating ~{expected_tokens} tokens...")
    print("  This may take several minutes on CPU. Please wait...")

    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max(expected_tokens + 200, 200),
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    print(f"\n{'=' * 60}")
    print("TRANSCRIPT (streaming):")
    print(f"{'=' * 60}")

    transcript = ""
    token_count = 0
    start_time = time.time()
    last_report = 0

    for new_text in streamer:
        transcript += new_text
        token_count += 1
        elapsed = time.time() - start_time

        if elapsed - last_report >= 2:
            tps = token_count / elapsed if elapsed > 0 else 0
            print(
                f"\r  [{elapsed:.0f} s elapsed | {token_count} tokens generated | {tps:.1f} t/s]",
                end="",
                flush=True,
            )
            last_report = elapsed

    thread.join()
    elapsed = time.time() - start_time
    tps = token_count / elapsed if elapsed > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"\nTranscription completed in {elapsed / 60:.1f} min ({tps:.1f} tokens/s)")
    print(f"Audio duration: {audio_dur:.1f}s")
    print(f"Real-time factor: {elapsed / audio_dur:.2f}x")
    print(f"Tokens generated: {token_count}")
    print(f"\n{'=' * 60}")
    print(f"TRANSCRIPT:")
    print(f"{'=' * 60}")
    print(transcript)
    print(f"{'=' * 60}")

    save_transcript(
        transcript,
        source=f"File: {audio_path.name}",
        start_time=datetime.datetime.now(),
        duration_sec=audio_dur,
        model_name=MODEL_NAME,
        model_params={"delay_ms": delay_ms} if delay_ms is not None else None,
    )

    return transcript


def transcribe_from_mic(model, processor, duration: int = 30, delay_ms: int = None, loopback: bool = False):
    """Record from microphone (or system audio loopback) and transcribe."""
    sample_rate = processor.feature_extractor.sampling_rate

    device = None
    if loopback:
        device = get_loopback_device()
        if device is None:
            print("Warning: WASAPI loopback device not found. Falling back to default mic.")
            source_label = "Microphone (loopback fallback)"
        else:
            print(f"Loopback device: {sd.query_devices(device)['name']}")
            source_label = "System Audio (loopback)"
    else:
        source_label = "Microphone"

    print(f"\nRecording for up to {duration} seconds... Speak now!")

    audio_data = []
    record_start = datetime.datetime.now()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}", file=sys.stderr)
        audio_data.append(indata.copy())

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
        callback=callback,
    ):
        record_until_esc_or_timeout(duration)

    if not audio_data:
        print("No audio recorded.")
        return

    audio_array = np.concatenate(audio_data, axis=0).flatten()
    audio_dur = len(audio_array) / sample_rate
    print(f"Recorded {audio_dur:.1f}s of audio")
    print("Processing...")

    inputs = processor(audio_array, return_tensors="pt")
    inputs = inputs.to(model.device, dtype=model.dtype)

    expected_tokens = int(audio_dur * 12.5)
    print(f"  Generating ~{expected_tokens} tokens...")

    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max(expected_tokens + 200, 200),
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    print(f"\n{'=' * 60}")
    print("TRANSCRIPT (streaming):")
    print(f"{'=' * 60}")

    transcript = ""
    token_count = 0
    start_time = time.time()
    last_report = 0

    for new_text in streamer:
        transcript += new_text
        token_count += 1
        elapsed = time.time() - start_time

        if elapsed - last_report >= 2:
            tps = token_count / elapsed if elapsed > 0 else 0
            print(
                f"\r  [{elapsed:.0f} s elapsed | {token_count} tokens generated | {tps:.1f} t/s]",
                end="",
                flush=True,
            )
            last_report = elapsed

    thread.join()
    elapsed = time.time() - start_time
    tps = token_count / elapsed if elapsed > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"\nTranscription completed in {elapsed / 60:.1f} min ({tps:.1f} tokens/s)")
    print(f"Audio duration: {audio_dur:.1f}s")
    print(f"Real-time factor: {elapsed / audio_dur:.2f}x")
    print(f"Tokens generated: {token_count}")
    print(f"\n{'=' * 60}")
    print(f"TRANSCRIPT:")
    print(f"{'=' * 60}")
    print(transcript)
    print(f"{'=' * 60}")

    save_transcript(
        transcript,
        source=source_label,
        start_time=record_start,
        duration_sec=audio_dur,
        model_name=MODEL_NAME,
        model_params={"delay_ms": delay_ms} if delay_ms is not None else None,
    )

    return transcript


def main():
    parser = argparse.ArgumentParser(
        description="Voxtral Mini 4B Realtime - Local CPU Transcription"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "audio_file", nargs="?", help="Path to audio file (mp3, wav, flac, etc.)"
    )
    group.add_argument(
        "--mic", action="store_true", help="Record from microphone and transcribe"
    )
    group.add_argument(
        "--mic-loop", action="store_true",
        help="Record system audio via WASAPI loopback (captures whatever is playing)"
    )
    parser.add_argument(
        "--model",
        default=VOXTRAL_MODEL_ID,
        help=f"Model ID or local path (default: {VOXTRAL_MODEL_ID})",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Custom HuggingFace cache directory for model weights",
    )
    parser.add_argument(
        "--dur",
        type=float,
        default=60,
        help="Recording duration in minutes when using --mic (default: 60)",
    )
    parser.add_argument(
        "--del",
        dest="delay",
        type=int,
        default=960,
        help="Transcription delay in ms (e.g. 480, 960, 2400). Higher = more accurate but more latency (default: 960)",
    )
    parser.add_argument(
        "--engine",
        dest="engine",
        type=str,
        default="voxtral",
        help="Engine to use: voxtral (default)",
    )
    args = parser.parse_args()

    if args.engine != "voxtral":
        print(
            f"Error: This script uses the Voxtral engine. Use 'python transcribe_voxtral.py' for Voxtral."
        )
        sys.exit(1)

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir

    model, processor = load_model_and_processor(args.model, args.delay)

    print(f"\n{'=' * 60}")
    print(f"  python transcribe_voxtral.py --MODE --DUR --DEL")
    print(f"  MODE:     mic  |  mic-loop  |  audio.mp3")
    print(f"  DURATION: (in min, default: 60)")
    print(f"  DELAY:    80, 480, 960, 2400  (in ms, default: 960)")
    print(f"\n  Press ESC to stop recording")
    print(f"{'=' * 60}\n")

    if args.mic or args.mic_loop:
        duration_sec = int(args.dur * 60)
        transcribe_from_mic(model, processor, duration_sec, delay_ms=args.delay, loopback=args.mic_loop)
    else:
        transcribe_audio(model, processor, args.audio_file, delay_ms=args.delay)


if __name__ == "__main__":
    main()
