"""
Speech-to-text transcription dispatcher.
Selects the model backend and forwards all remaining arguments to it.

Usage:
    python transcribe.py --voxtral --mic --dur 60 --del 960
    python transcribe.py --parakeet --mic --dur 30
    python transcribe.py --voxtral audio.mp3
    python transcribe.py --parakeet audio.mp3
    python transcribe.py --help
"""

import subprocess
import sys


def main():
    args = sys.argv[1:]

    if "--voxtral" in args:
        args.remove("--voxtral")
        script = "transcribe_voxtral.py"
    elif "--parakeet" in args:
        args.remove("--parakeet")
        script = "transcribe_parakeet.py"
    else:
        print("Error: specify a model with --voxtral or --parakeet")
        print()
        print("Usage:")
        print("  python transcribe.py --voxtral  --mic --dur 60 --del 960")
        print("  python transcribe.py --parakeet --mic --dur 60")
        print("  python transcribe.py --voxtral  audio.mp3")
        print("  python transcribe.py --parakeet audio.mp3")
        sys.exit(1)

    cmd = [sys.executable, script] + args
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
