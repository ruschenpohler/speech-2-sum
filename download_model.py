"""
Download the Voxtral model weights locally for offline use.
Run this once - weights are cached for all future transcriptions.

Usage:
    python download_model.py
    python download_model.py --cache-dir D:\models
"""

import argparse
import os
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download Voxtral Mini 4B Realtime model"
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
    print(f"Downloading {model_id}...")
    print(f"This is ~8GB and may take a while depending on your connection.")

    path = snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.safetensors", "*.json", "*.model"],
    )

    print(f"\nModel downloaded to: {path}")
    print("You can now run transcriptions offline with:")
    print(
        f"  python transcribe.py audio.mp3 --cache-dir {args.cache_dir or ''}".strip()
    )


if __name__ == "__main__":
    main()
