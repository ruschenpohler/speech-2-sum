"""
Speech-to-text transcription dispatcher.
Selects the model backend and forwards all remaining arguments to it.
Optionally auto-summarizes the transcript after recording completes.

Usage:
    python transcribe.py --voxtral  --mic --dur 60 --del 960
    python transcribe.py --parakeet --mic --dur 30
    python transcribe.py --voxtral  --mic --sum
    python transcribe.py --voxtral  audio.mp3
    python transcribe.py --parakeet audio.mp3
    python transcribe.py --help
"""

import subprocess
import sys
from pathlib import Path


def main():
    args = sys.argv[1:]

    # Extract --sum before forwarding args to transcription script
    run_sum = "--sum" in args
    if run_sum:
        args.remove("--sum")

    # Extract --keep / --max-cont / --max-out for summarizer, leave rest for transcriber
    sum_args = []
    for flag in ("--keep", "--max-cont", "--max-out"):
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args):
                sum_args += [flag, args[idx + 1]]
                args.pop(idx + 1)
                args.pop(idx)

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

    result = subprocess.run([sys.executable, script] + args)

    if run_sum and result.returncode == 0:
        transcripts_dir = Path("transcripts")
        if transcripts_dir.exists():
            candidates = list(transcripts_dir.glob("*.md"))
            if candidates:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                print(f"\n[Auto-summarization] Transcript: {latest.name}")
                subprocess.run(
                    [sys.executable, "summarize.py", str(latest)] + sum_args
                )

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
