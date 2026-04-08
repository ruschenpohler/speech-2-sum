"""
Download the Voxtral Mini 4B Realtime model weights locally for offline use.
Run this once - weights are cached for all future transcriptions.

Usage:
    python download_voxtral.py
    python download_voxtral.py --cache-dir D:\models
"""

import argparse
import os
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download Voxtral Mini 4B Realtime model weights"
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Custom HuggingFace cache directory (default: ~/.cache/huggingface)",
    )
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir

    model_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"
    print(f"Downloading Voxtral: {model_id}")
    print("This is ~8GB and may take a while depending on your connection.")

    path = snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.safetensors", "*.json", "*.model"],
    )

    print(f"\nVoxtral downloaded to: {path}")
    print("You can now run Voxtral transcriptions offline with:")
    print(
        f"  python transcribe_voxtral.py audio.mp3 --cache-dir {args.cache_dir or ''}".strip()
    )


if __name__ == "__main__":
    main()
