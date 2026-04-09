"""
Download Qwen2.5-7B-Instruct GGUF model for local summarization.
Run once — weights are cached for all future summaries.

Usage:
    python download_summarizer.py
    python download_summarizer.py --cache-dir D:\\models
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import hf_hub_download


MODEL_REPO = "second-state/Qwen2.5-7B-Instruct-GGUF"
MODEL_FILE = "Qwen2.5-7B-Instruct-Q8_0.gguf"


def main():
    parser = argparse.ArgumentParser(
        description="Download Qwen2.5-7B-Instruct GGUF model for summarization"
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to save the model file (default: ./models)",
    )
    args = parser.parse_args()

    local_dir = Path(args.cache_dir) if args.cache_dir else Path("models")
    local_dir.mkdir(parents=True, exist_ok=True)

    dest = local_dir / MODEL_FILE
    if dest.exists():
        print(f"Model already exists at: {dest}")
        print("Delete it manually if you want to re-download.")
        return

    print(f"Downloading {MODEL_REPO}/{MODEL_FILE}")
    print(f"Destination: {local_dir}")
    print("This is ~6.7 GB and may take a while...\n")

    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=str(local_dir),
    )

    print(f"\nDownloaded to: {path}")
    print("\nYou can now run summarization with:")
    print(f"  python summarize.py transcripts/your_file.md")


if __name__ == "__main__":
    main()
