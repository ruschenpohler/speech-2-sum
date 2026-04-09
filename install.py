"""
Interactive setup: creates a uv virtual environment and installs the
model backend(s) of your choice.

Usage:
    python install.py
"""

import subprocess
import sys


VOXTRAL_INFO = """\
  Voxtral Mini 4B Realtime (Mistral AI / HuggingFace Transformers)
    - ~8 GB download, ~8 GB RAM during inference
    - Streaming output (see words appear as they're transcribed)
    - Tunable transcription delay (accuracy vs. latency trade-off)
    - Works on native Windows
"""

PARAKEET_INFO = """\
  Parakeet TDT 0.6B (NVIDIA / NeMo)
    - ~2.5 GB download, ~1.5 GB RAM during inference
    - Batch processing (faster on CPU, result appears all at once)
    - No delay tuning needed
    - Officially supported on Linux/macOS; on Windows, WSL2 is recommended
"""

SUMMARIZE_INFO = """\
  Qwen2.5-7B Summarization (llama.cpp / GGUF Q8_0)
    - ~6.7 GB download, ~12 GB RAM during inference
    - Local abstractive summarization with key quote extraction
    - Runs on CPU (no GPU required)
    - Works on native Windows
"""


def ask(prompt, valid):
    while True:
        ans = input(prompt).strip().lower()
        if ans in valid:
            return ans
        print(f"  Please enter one of: {', '.join(valid)}")


def run(cmd, **kwargs):
    print(f"\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"\nError: command failed with exit code {result.returncode}.")
        sys.exit(result.returncode)


def main():
    print("\n=== speech-2-sum setup ===\n")
    print("This will create a virtual environment and install dependencies.\n")
    print("Model backends and extras available:\n")
    print(VOXTRAL_INFO)
    print(PARAKEET_INFO)
    print(SUMMARIZE_INFO)

    print("Which backend(s) do you want to install?")
    print("  [v] Voxtral only")
    print("  [p] Parakeet only")
    print("  [s] Summarization only")
    print("  [b] Voxtral + Parakeet")
    print("  [a] All three\n")

    choice = ask("Your choice (v/p/s/b/a): ", {"v", "p", "s", "b", "a"})
    extras_map = {
        "v": "voxtral",
        "p": "parakeet",
        "s": "summarize",
        "b": "voxtral,parakeet",
        "a": "all",
    }
    extras = extras_map[choice]

    if choice in ("p", "b", "a"):
        print(
            "\nNote: Parakeet uses NVIDIA NeMo, which is not officially supported on "
            "native Windows.\nIf you're on Windows, running inside WSL2 is recommended."
        )
        cont = ask("Continue anyway? (y/n): ", {"y", "n"})
        if cont == "n":
            print("Aborted. Re-run install.py and choose Voxtral only, or switch to WSL2.")
            sys.exit(0)

    print(f"\nCreating virtual environment with uv...")
    run(["uv", "venv", ".venv"])

    print(f"\nInstalling dependencies for: {extras}...")
    run(["uv", "pip", "install", "-e", f".[{extras}]"])

    print(f"\n{'=' * 50}")
    print(f"  Setup complete! Activate your environment with:")
    print(f"    .venv\\Scripts\\Activate.ps1      (PowerShell)")
    print(f"    source .venv/bin/activate        (bash/zsh)")
    print(f"\n  Then download model weights:")
    if choice in ("v", "b", "a"):
        print(f"    python download_voxtral.py")
    if choice in ("p", "b", "a"):
        print(f"    python download_parakeet.py")
    if choice in ("s", "a"):
        print(f"    python download_summarizer.py")
    print(f"\n  Then transcribe:")
    if choice in ("v", "b", "a"):
        print(f"    python transcribe.py --voxtral --mic")
    if choice in ("p", "b", "a"):
        print(f"    python transcribe.py --parakeet --mic")
    if choice in ("s", "a"):
        print(f"    python summarize.py transcripts/your_file.md")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
