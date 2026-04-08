"""
Download the NVIDIA Parakeet TDT 0.6B model weights locally for offline use.
Run this once - weights are cached for all future transcriptions.

Usage:
    python download_parakeet.py
    python download_parakeet.py --cache-dir D:\\models
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Download NVIDIA Parakeet TDT 0.6B model weights"
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Custom NeMo cache directory (default: ~/.cache/torch/NeMo)",
    )
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["NEMO_CACHE_DIR"] = args.cache_dir

    model_id = "nvidia/parakeet-tdt-0.6b-v3"
    print(f"Downloading Parakeet: {model_id}")
    print("This is ~2.5GB and may take a while depending on your connection.")
    print("NeMo will handle caching automatically.\n")

    import nemo.collections.asr as nemo_asr

    print("Loading model (this triggers the download on first run)...")
    model = nemo_asr.models.ASRModel.from_pretrained(model_id=model_id)

    cache_note = f" at {args.cache_dir}" if args.cache_dir else ""
    print(f"\nParakeet downloaded and cached{cache_note}.")
    print("You can now run Parakeet transcriptions offline with:")
    cache_arg = f" --cache-dir {args.cache_dir}" if args.cache_dir else ""
    print(f"  python transcribe.py --parakeet --mic{cache_arg}")
    print(f"  python transcribe_parakeet.py --mic{cache_arg}")


if __name__ == "__main__":
    main()
