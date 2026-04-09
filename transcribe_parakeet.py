"""
NVIDIA Parakeet TDT 0.6B - CPU Transcription Script
Uses NVIDIA NeMo for local, private speech-to-text.
All processing happens locally - no data leaves your machine.

Parakeet is batch-only (no streaming), but substantially faster than
Voxtral on CPU due to its smaller parameter count.

Usage:
    python transcribe_parakeet.py audio_file.mp3
    python transcribe_parakeet.py audio_file.wav
    python transcribe_parakeet.py --mic
    python transcribe_parakeet.py --help
"""

import argparse
import datetime
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from utils import print_ram, save_transcript, record_until_esc_or_timeout, get_loopback_device

MODEL_NAME = "Parakeet"
PARAKEET_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"


def load_model(model_id: str = PARAKEET_MODEL_ID):
    """Load the Parakeet ASR model on CPU."""
    print_ram("Before loading")

    print(f"\n[Step 1/2] Importing NeMo ASR toolkit...")
    import nemo.collections.asr as nemo_asr

    print("[Step 1/2] NeMo imported.")

    print(f"\n[Step 2/2] Loading Parakeet model from {model_id}...")
    print("  This may take a minute on first run. Please wait...")
    t0 = time.time()
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
    model.eval()
    elapsed = time.time() - t0
    print(f"[Step 2/2] Parakeet model ready in {elapsed:.0f}s.")
    print_ram("After initialization")

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    weight_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    print(f"\n{'=' * 40}")
    print(f"  Model:       Parakeet TDT 0.6B v3")
    print(f"  Parameters:  {param_count:.1f}B")
    print(f"  Weights RAM: ~{weight_mem:.2f}GB (FP32)")
    print(f"{'=' * 40}")

    return model


def transcribe_audio(model, audio_path: str):
    """Transcribe a single audio file."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    print(f"\nLoading audio: {audio_path.name}")

    # NeMo expects a file path; use it directly if wav, otherwise convert via soundfile
    audio_data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)  # stereo → mono
    audio_dur = len(audio_data) / sample_rate
    print(f"  Duration: {audio_dur:.1f}s | Sample rate: {sample_rate}Hz")

    print(f"\nTranscribing with Parakeet (batch mode)...")
    print(
        "  Parakeet processes the full audio in one pass - much faster than real-time."
    )

    t0 = time.time()
    # NeMo transcribe() accepts file paths; write to a temp wav if needed
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, audio_data, sample_rate)

    results = model.transcribe([tmp_path])
    Path(tmp_path).unlink(missing_ok=True)

    elapsed = time.time() - t0
    transcript = results[0] if isinstance(results[0], str) else results[0].text

    print(f"\n{'=' * 60}")
    print(f"\nTranscription completed in {elapsed:.1f}s")
    print(f"Audio duration: {audio_dur:.1f}s")
    print(f"Real-time factor: {elapsed / audio_dur:.2f}x")
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
    )

    return transcript


def transcribe_from_mic(model, duration: int = 30, loopback: bool = False):
    """Record from microphone (or system audio loopback) and transcribe."""
    sample_rate = 16000  # Parakeet expects 16kHz

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

    print(f"\nTranscribing with Parakeet (batch mode)...")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, audio_array, sample_rate)

    t0 = time.time()
    results = model.transcribe([tmp_path])
    Path(tmp_path).unlink(missing_ok=True)
    elapsed = time.time() - t0

    transcript = results[0] if isinstance(results[0], str) else results[0].text

    print(f"\n{'=' * 60}")
    print(f"\nTranscription completed in {elapsed:.1f}s")
    print(f"Audio duration: {audio_dur:.1f}s")
    print(f"Real-time factor: {elapsed / audio_dur:.2f}x")
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
    )

    return transcript


def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA Parakeet TDT 0.6B - Local CPU Transcription"
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
        default=PARAKEET_MODEL_ID,
        help=f"NeMo model ID or local path (default: {PARAKEET_MODEL_ID})",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Custom NeMo cache directory for model weights",
    )
    parser.add_argument(
        "--dur",
        type=float,
        default=60,
        help="Recording duration in minutes when using --mic (default: 60)",
    )
    # --del is a no-op for Parakeet (Voxtral-specific feature), kept for CLI consistency
    parser.add_argument(
        "--del",
        dest="delay",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # hidden: not applicable to Parakeet
    )
    args = parser.parse_args()

    if args.delay is not None:
        print(
            "Note: --del (transcription delay) is a Voxtral-only option and has no effect on Parakeet."
        )

    if args.cache_dir:
        import os

        os.environ["NEMO_CACHE_DIR"] = args.cache_dir

    model = load_model(args.model)

    print(f"\n{'=' * 60}")
    print(f"  python transcribe_parakeet.py --MODE --DUR")
    print(f"  MODE:     mic  |  mic-loop  |  audio.mp3")
    print(f"  DURATION: (in min, default: 60)")
    print(f"\n  Press ESC to stop recording")
    print(f"{'=' * 60}\n")

    if args.mic or args.mic_loop:
        duration_sec = int(args.dur * 60)
        transcribe_from_mic(model, duration_sec, loopback=args.mic_loop)
    else:
        transcribe_audio(model, args.audio_file)


if __name__ == "__main__":
    main()
