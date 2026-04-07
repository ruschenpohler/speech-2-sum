"""
Voxtral Mini 4B Realtime - CPU Transcription Script
Uses HuggingFace Transformers for local, private speech-to-text.
All processing happens locally - no data leaves your machine.

Usage:
    python transcribe.py audio_file.mp3
    python transcribe.py audio_file.wav
    python transcribe.py --mic
    python transcribe.py --help
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import sounddevice as sd
import torch
import urllib.request
from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
from mistral_common.tokens.tokenizers.audio import Audio


def print_ram(label=""):
    """Print current process RAM usage."""
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    total = psutil.virtual_memory()
    prefix = f"[RAM] {label} " if label else "[RAM] "
    print(
        f"{prefix}Process: {mem.rss / 1024**3:.1f}GB | "
        f"System: {total.used / 1024**3:.1f}GB / {total.total / 1024**3:.1f}GB "
        f"({total.percent}% used)"
    )


def load_model_and_processor(
    model_id: str = "mistralai/Voxtral-Mini-4B-Realtime-2602",
    delay_ms: int = None,
):
    """Load the Voxtral model and processor on CPU."""
    print_ram("Before loading")

    if delay_ms is not None:
        print(f"\n  Setting transcription delay to {delay_ms}ms...")
        _set_transcription_delay(model_id, delay_ms)

    print(f"\n[Step 1/3] Loading tokenizer from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    print("[Step 1/3] Tokenizer loaded.")
    print_ram("After tokenizer")

    print("\n[Step 2/3] Loading model weights into CPU memory...")
    print("  This takes ~1-2 minutes on CPU. Please wait...")
    t0 = time.time()
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    elapsed = time.time() - t0
    print(f"[Step 2/3] Model weights loaded in {elapsed:.0f}s.")
    print_ram("After model weights")

    print("\n[Step 3/3] Initializing model on CPU...")
    t0 = time.time()
    _ = model.device
    elapsed = time.time() - t0
    print(f"[Step 3/3] Model ready in {elapsed:.0f}s.")
    print_ram("After initialization")

    weight_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    print(f"\n{'=' * 40}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    print(f"  Weights RAM: ~{weight_mem:.1f}GB (BF16)")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype: {next(model.parameters()).dtype}")
    print(f"{'=' * 40}")

    return model, processor


def _set_transcription_delay(model_id: str, delay_ms: int):
    """Modify transcription_delay_ms in the local model cache."""
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(repo_id=model_id)
    tekken_path = os.path.join(model_dir, "tekken.json")

    import json

    with open(tekken_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["config"]["transcription_delay_ms"] = delay_ms

    with open(tekken_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"  Transcription delay set to {delay_ms}ms in {tekken_path}")


def transcribe_audio(model, processor, audio_path: str):
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

    start_time = time.time()
    outputs = model.generate(**inputs)
    elapsed = time.time() - start_time

    decoded = processor.batch_decode(outputs, skip_special_tokens=True)
    transcript = decoded[0]

    print(f"\nTranscription completed in {elapsed:.1f}s")
    print(
        f"Audio duration: {len(audio.audio_array) / processor.feature_extractor.sampling_rate:.1f}s"
    )
    print(
        f"Real-time factor: {elapsed / (len(audio.audio_array) / processor.feature_extractor.sampling_rate):.2f}x"
    )
    print(f"\n{'=' * 60}")
    print(f"TRANSCRIPT:")
    print(f"{'=' * 60}")
    print(transcript)
    print(f"{'=' * 60}")

    save_transcript(transcript, source=f"File: {audio_path.name}")

    return transcript


def get_wifi_name():
    """Get current WiFi network name on Windows."""
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            if "SSID" in line and "BSSID" not in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "Unknown"


def get_location():
    """Get approximate location from public IP geolocation."""
    try:
        req = urllib.request.Request(
            "http://ip-api.com/json/", headers={"User-Agent": "transcribe"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        parts = []
        for key in ["city", "regionName", "country"]:
            if data.get(key):
                parts.append(data[key])
        return ", ".join(parts) if parts else "Unknown"
    except Exception:
        return "Unknown"


def save_transcript(transcript: str, source: str):
    """Save transcript to transcripts/ folder as timestamped .md file."""
    out_dir = Path("transcripts")
    out_dir.mkdir(exist_ok=True)

    now = datetime.datetime.now()
    ts = now.strftime("%Y-%m-%d_%H-%M-%S")
    wifi = get_wifi_name()
    location = get_location()

    filename = out_dir / f"{ts}.md"
    header = f"# {now.strftime('%Y-%m-%d %H:%M:%S')} | {location} | WiFi: {wifi} | Source: {source}"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(header + "\n\n")
        f.write(transcript + "\n")

    print(f"\nSaved to: {filename}")
    return filename


def transcribe_from_mic(model, processor, duration: int = 30):
    """Record from microphone and transcribe."""
    sample_rate = processor.feature_extractor.sampling_rate

    print(f"\nRecording for {duration} seconds... Speak now!")
    print("Press Ctrl+C to stop early.\n")

    audio_data = []

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}", file=sys.stderr)
        audio_data.append(indata.copy())

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            time.sleep(duration)
    except KeyboardInterrupt:
        print("\nRecording stopped.")

    if not audio_data:
        print("No audio recorded.")
        return

    audio_array = np.concatenate(audio_data, axis=0).flatten()
    print(f"Recorded {len(audio_array) / sample_rate:.1f}s of audio")
    print("Processing...")

    inputs = processor(audio_array, return_tensors="pt")
    inputs = inputs.to(model.device, dtype=model.dtype)

    start_time = time.time()
    outputs = model.generate(**inputs)
    elapsed = time.time() - start_time

    decoded = processor.batch_decode(outputs, skip_special_tokens=True)
    transcript = decoded[0]

    print(f"\nTranscription completed in {elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"TRANSCRIPT:")
    print(f"{'=' * 60}")
    print(transcript)
    print(f"{'=' * 60}")

    save_transcript(transcript, source="Microphone")

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
    parser.add_argument(
        "--model",
        default="mistralai/Voxtral-Mini-4B-Realtime-2602",
        help="Model ID or local path (default: mistralai/Voxtral-Mini-4B-Realtime-2602)",
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
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir

    model, processor = load_model_and_processor(args.model, args.delay)

    print(f"\n{'=' * 60}")
    print(f"  python transcribe.py --MODE --DUR --DEL")
    print(f"  MODE:     mic  |  audio.mp3")
    print(f"  DURATION: (in min, default: 60)")
    print(f"  DELAY:    80, 480, 960, 2400  (in ms, default: 960)")
    print(f"\n  To end transcription, press CTRL + C")
    print(f"{'=' * 60}\n")

    if args.mic:
        duration_sec = int(args.dur * 60)
        transcribe_from_mic(model, processor, duration_sec)
    else:
        transcribe_audio(model, processor, args.audio_file)


if __name__ == "__main__":
    main()
