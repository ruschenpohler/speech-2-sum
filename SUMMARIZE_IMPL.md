# Summarization Pipeline Implementation Plan
# Qwen2.5-7B (llama.cpp, GGUF 8-bit) - CPU Inference on DRAM

## Overview

Add a local text summarization feature that:
- Uses Qwen2.5-7B-Instruct (GGUF Q8_0, 8-bit quantized)
- Runs via llama.cpp on CPU (DRAM only, ~12GB total)
- Produces **hybrid output**: abstractive summary + extractive quotes
- Saves to `summaries/[transcription_filename]_SUM.md`

---

## File Structure (Post-Implementation)

```
s2t/
├── .venv/                          # Shared virtual environment
├── utils.py                        # Shared: print_ram, get_wifi_name, get_location, save_transcript
├── transcribe.py                   # Dispatcher: --voxtral, --parakeet
├── transcribe_voxtral.py           # Voxtral transcription
├── transcribe_parakeet.py          # Parakeet transcription
├── summarize.py                   # NEW: Qwen summarization script
├── download_summarizer.py          # NEW: Download Qwen GGUF model
├── SUMMARIZE_IMPL.md              # This file
├── README.md                       # Updated with summarization docs
├── pyproject.toml                  # Updated with summarization deps
├── transcripts/                    # Existing: transcription outputs
│   └── 2026-04-08_14h-32m--5min.md
└── summaries/                      # NEW: summarization outputs
    └── 2026-04-08_14h-32m--5min_SUM.md
```

---

## Step 1: Dependencies & Setup

### Update `pyproject.toml`

Add llama.cpp bindings (python bindings via `llama-cpp-python`):

```toml
[project.optional-dependencies]
# ... existing ...

# Qwen2.5-7B Summarization (llama.cpp)
summarize = [
    "llama-cpp-python>=0.2.0",
]
```

### Install

```powershell
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[summarize]"
```

---

## Step 2: Download Qwen GGUF (Q8_0)

### Option A: Pre-converted (Recommended)

Download from TheBespokeAI's collection on HuggingFace:

```powershell
# Direct download URL
# https://huggingface.co/TheBespokeAI/Qwen2.5-7B-Instruct-GGUF
# File: qwen2.5-7b-instruct-q8_0.gguf (~6.7 GB)

# Using huggingface-cli (recommended):
huggingface-cli download TheBespokeAI/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q8_0.gguf --local-dir models/

# Alternative: Use download_summarizer.py (see Step 3)
python download_summarizer.py
```

### Option B: Other GGUF Sources

If TheBespokeAI is unavailable, try:
- `https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF`
- `https://huggingface.co/Qwen/Qwen2.5-7B-Instruct`

### Model Specifications

| Property | Value |
|----------|-------|
| Model | Qwen2.5-7B-Instruct |
| Format | GGUF (llama.cpp native) |
| Quantization | Q8_0 (8-bit) |
| Size | ~6.7 GB |
| Context | 32K (native), truncate to MAX_CONT for CPU |
| RAM usage | ~8 GB weights + ~4 GB KV cache = ~12 GB total |

---

## Step 3: download_summarizer.py

Create `download_summarizer.py` to handle model download:

```python
"""
Download Qwen2.5-7B-Instruct GGUF model for local summarization.
Run once - weights are cached for all future summaries.

Usage:
    python download_summarizer.py
    python download_summarizer.py --cache-dir D:\models
    python download_summarizer.py --model qwen2.5-7b-instruct-q8_0.gguf
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

MODEL_REPO = "TheBespokeAI/Qwen2.5-7B-Instruct-GGUF"
DEFAULT_FILE = "qwen2.5-7b-instruct-q8_0.gguf"


def main():
    parser = argparse.ArgumentParser(description="Download Qwen2.5-7B GGUF model for summarization")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Custom cache directory (default: ./models)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_FILE,
        help=f"GGUF filename to download (default: {DEFAULT_FILE})",
    )
    args = parser.parse_args()

    local_dir = args.cache_dir or os.path.join(os.getcwd(), "models")
    os.makedirs(local_dir, exist_ok=True)

    print(f"Downloading {MODEL_REPO}/{args.model}")
    print(f"Destination: {local_dir}")
    print("This is ~6.7 GB and may take a while...")

    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=args.model,
        local_dir=local_dir,
    )

    print(f"\nDownloaded to: {path}")
    print("\nYou can now run summarization with:")
    print(f"  python summarize.py --qwen transcripts/your_file.md")


if __name__ == "__main__":
    main()
```

---

## Step 4: summarize.py Implementation

### CLI Interface

```powershell
# Standalone usage
python summarize.py --qwen transcripts/2026-04-08_14h-32m--5min.md
python summarize.py --qwen --max-cont 2000

# Via dispatcher (in transcribe.py)
python transcribe.py --voxtral --mic --sum
python transcribe.py --parakeet --mic --sum
```

### Key Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--qwen` | required | Model engine selector |
| `input_file` | required | Path to transcription .md file |
| `--max-cont` | 4000 | Max context tokens (input truncation) |
| `--cache-dir` | optional | Custom model path |

### Internal Constants

```python
# Model configuration
MODEL_PATH = "models/qwen2.5-7b-instruct-q8_0.gguf"  # or --cache-dir override
N_GPU_LAYERS = 0          # CPU-only (0 = use CPU)
N_THREADS = os.cpu_count()  # Auto-detect CPU cores
BATCH_SIZE = 1             # Hardcoded (as requested)
MAX_TOKENS = 500           # Max output tokens for summary
TEMPERATURE = 0.3          # Low for deterministic output
```

### Full Implementation

```python
"""
Qwen2.5-7B Summarization - Local CPU Inference
Uses llama.cpp (GGUF Q8_0) for efficient CPU-only inference.

Usage:
    python summarize.py --qwen transcripts/2026-04-08_14h-32m--5min.md
    python summarize.py --qwen --max-cont 2000 transcripts/file.md
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil

# llama-cpp-python bindings
from llama_cpp import Llama


# =============================================================================
# Shared utilities (reused from utils.py)
# =============================================================================

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
    """Get approximate location and timezone from public IP geolocation."""
    try:
        req = urllib.request.Request(
            "http://ip-api.com/json/", headers={"User-Agent": "summarize"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        parts = []
        for key in ["city", "regionName", "country"]:
            if data.get(key):
                parts.append(data[key])
        location = ", ".join(parts) if parts else "Unknown"
        timezone = data.get("timezone", "")
        return location, timezone
    except Exception:
        return "Unknown", ""


# =============================================================================
# Summarization prompt
# =============================================================================

SUMMARIZE_PROMPT = """You are a professional summarization assistant. Analyze the following transcription and provide:

1. SUMMARY: A concise abstractive summary (2-4 sentences) that captures the key points.
2. KEY QUOTES: 3-5 verbatim extracts from the text that best represent the content.
   - Each quote must be a direct extract, enclosed in quotation marks
   - Include context: speaker name or "[Speaker N]" if unidentified

Format your response as:

SUMMARY:
[Your 2-4 sentence summary here]

KEY QUOTES:
- "[First exact quote from text]"
- "[Second exact quote]"
- "[Third exact quote]"
...

TRANSCRIPTION:
{input_text}"""


# =============================================================================
# Model loading
# =============================================================================

def load_model(model_path: str, n_threads: int = None):
    """Load Qwen2.5-7B GGUF model via llama.cpp."""
    if n_threads is None:
        n_threads = os.cpu_count() or 4

    print(f"\n[Step 1/2] Loading model from {model_path}...")
    print(f"  Using {n_threads} CPU threads (CPU-only, no GPU)")

    t0 = time.time()

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=0,          # CPU only
        n_threads=n_threads,
        n_ctx=4096,               # Context window (truncate input if longer)
        verbose=False,
    )

    elapsed = time.time() - t0
    print(f"[Step 1/2] Model loaded in {elapsed:.1f}s")

    return llm


# =============================================================================
# Summarization
# =============================================================================

def summarize_text(
    llm,
    input_file: str,
    max_context: int = 4000,
    max_tokens: int = 500,
    temperature: float = 0.3,
):
    """Generate summary + extract quotes from transcription."""

    # Read input file
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract body (skip header lines starting with #)
    lines = content.split("\n")
    body_lines = []
    in_header = True
    for line in lines:
        if in_header and line.startswith("#"):
            continue
        if in_header and line.strip() == "":
            in_header = False
            continue
        if not in_header:
            body_lines.append(line)

    transcription = "\n".join(body_lines).strip()

    # Estimate tokens (rough: 1 token ≈ 4 chars for English)
    estimated_tokens = len(transcription) // 4

    # Truncate if needed
    if estimated_tokens > max_context:
        print(f"  Input ~{estimated_tokens} tokens, truncating to {max_context}")
        # Keep last max_context tokens (more likely to have recent content)
        chars_to_keep = max_context * 4
        transcription = transcription[-chars_to_keep:]

    # Build prompt
    prompt = SUMMARIZE_PROMPT.format(input_text=transcription)

    print(f"\n[Step 2/2] Generating summary...")
    print(f"  Input: ~{len(transcription) // 4} tokens")
    print(f"  Max output: {max_tokens} tokens")

    t0 = time.time()

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["TRANSCRIPTION:", "\n\n\n"],
    )

    elapsed = time.time() - t0
    generated_text = output["choices"][0]["text"].strip()

    print(f"  Generated in {elapsed:.1f}s (~{len(generated_text) // 4} tokens)")
    print(f"  Output tokens/second: {len(generated_text) // 4 / elapsed:.1f}")

    return generated_text, len(transcription) // 4, len(generated_text) // 4, elapsed


def parse_output(output_text: str):
    """Parse the LLM output into summary and quotes."""

    summary_match = re.search(r"SUMMARY:\s*(.+?)(?=KEY QUOTES:|$)", output_text, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else "No summary generated"

    quotes_match = re.search(r"KEY QUOTES:\s*(.+)$", output_text, re.DOTALL)
    quotes_text = quotes_match.group(1).strip() if quotes_match else ""

    # Extract quoted strings
    quotes = re.findall(r'- "?([^"]+)"?', quotes_text)
    if not quotes:
        quotes = re.findall(r'"([^"]+)"', quotes_text)

    return summary, quotes


# =============================================================================
# Save output
# =============================================================================

def save_summary(
    summary: str,
    quotes: list,
    input_file: str,
    input_tokens: int,
    output_tokens: int,
    elapsed: float,
):
    """Save summary to summaries/ folder."""

    input_path = Path(input_file)
    output_dir = Path("summaries")
    output_dir.mkdir(exist_ok=True)

    # Filename: [transcription_name]_SUM.md
    output_filename = input_path.stem + "_SUM.md"
    output_path = output_dir / output_filename

    # Get metadata
    now = datetime.datetime.now()
    wifi = get_wifi_name()
    location, timezone = get_location()

    ts_with_tz = now.strftime("%Y-%m-%d %H:%M:%S")
    if timezone:
        ts_with_tz += f" {timezone}"

    # Build content
    content = f"""# Summary: {input_path.stem}

## Model: Qwen2.5-7B-Instruct (GGUF Q8_0, 8-bit)
## Context: {input_tokens} tokens (truncated from input)
## Generated: {ts_with_tz}
## Location: {location}
## WiFi: {wifi}

---

### SUMMARY

{summary}

---

### KEY QUOTES

"""
    for i, quote in enumerate(quotes, 1):
        content += f"- \"{quote}\"\n"

    content += f"""

---

### METRICS
- Input tokens: {input_tokens}
- Output tokens: {output_tokens}
- Processing time: {elapsed:.1f}s
- Model: Qwen2.5-7B-Instruct (llama.cpp, CPU-only)
- Quantization: Q8_0 (8-bit)
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nSaved to: {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-7B Summarization - Local CPU Inference"
    )
    parser.add_argument(
        "--qwen",
        action="store_true",
        help="Use Qwen2.5-7B model (required)",
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Path to transcription .md file to summarize",
    )
    parser.add_argument(
        "--max-cont",
        type=int,
        default=4000,
        help="Max context tokens (default: 4000)",
    )
    parser.add_argument(
        "--max-out",
        type=int,
        default=500,
        help="Max output tokens (default: 500)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Custom model cache directory",
    )
    args = parser.parse_args()

    # Validate
    if not args.qwen:
        print("Error: --qwen flag is required")
        print("Usage: python summarize.py --qwen transcripts/file.md")
        sys.exit(1)

    if not args.input_file:
        print("Error: input_file is required")
        print("Usage: python summarize.py --qwen transcripts/file.md")
        sys.exit(1)

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    # Model path
    if args.cache_dir:
        model_dir = args.cache_dir
    else:
        model_dir = os.path.join(os.getcwd(), "models")

    model_path = os.path.join(model_dir, "qwen2.5-7b-instruct-q8_0.gguf")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Run: python download_summarizer.py")
        sys.exit(1)

    # Show RAM before loading
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    total = psutil.virtual_memory()
    print(f"[RAM] Before loading: {mem.rss / 1024**3:.1f}GB process | {total.percent}% system")

    # Load model
    llm = load_model(model_path)

    # Show RAM after loading
    mem = proc.memory_info()
    print(f"[RAM] After model: {mem.rss / 1024**3:.1f}GB process | {total.percent}% system")

    # Generate summary
    output_text, input_tokens, output_tokens, elapsed = summarize_text(
        llm,
        args.input_file,
        max_context=args.max_cont,
        max_tokens=args.max_out,
    )

    # Parse output
    summary, quotes = parse_output(output_text)

    # Print to console
    print(f"\n{'=' * 60}")
    print("SUMMARY:")
    print(f"{'=' * 60}")
    print(summary)
    print(f"\n{'=' * 60}")
    print("KEY QUOTES:")
    print(f"{'=' * 60}")
    for quote in quotes:
        print(f"- \"{quote}\"")
    print(f"{'=' * 60}")

    # Save to file
    save_summary(
        summary,
        quotes,
        args.input_file,
        input_tokens,
        output_tokens,
        elapsed,
    )


if __name__ == "__main__":
    main()
```

---

## Step 5: Integration with transcribe.py

Modify `transcribe.py` dispatcher to accept `--sum` flag:

```python
# In transcribe.py dispatcher
import subprocess
import sys
import os

def main():
    args = sys.argv[1:]

    # Handle model selection
    run_summarize = False
    summarize_file = None

    if "--sum" in args:
        args.remove("--sum")
        run_summarize = True

    if "--voxtral" in args:
        args.remove("--voxtral")
        script = "transcribe_voxtral.py"
    elif "--parakeet" in args:
        args.remove("--parakeet")
        script = "transcribe_parakeet.py"
    else:
        print("Error: specify a model with --voxtral or --parakeet")
        sys.exit(1)

    # Run transcription
    cmd = [sys.executable, script] + args
    result = subprocess.run(cmd)

    # If --sum was passed, run summarization on the output
    if run_summarize and result.returncode == 0:
        # Find the most recent transcript file
        transcripts_dir = Path("transcripts")
        if transcripts_dir.exists():
            latest = max(transcripts_dir.glob("*.md"), key=lambda p: p.stat().st_mtime)
            print(f"\n[Auto-summarization] Found: {latest.name}")
            print("Running summarization...\n")

            summarize_cmd = [
                sys.executable, "summarize.py",
                "--qwen",
                str(latest),
            ]
            # Pass through max-cont if user specified it
            if "--max-cont" in args:
                idx = args.index("--max-cont")
                if idx + 1 < len(args):
                    summarize_cmd.extend(["--max-cont", args[idx + 1]])

            subprocess.run(summarize_cmd)

    sys.exit(result.returncode)
```

### Workflow

1. User runs: `python transcribe.py --voxtral --mic --sum`
2. Transcription runs → saves to `transcripts/...md`
3. User presses Ctrl+C to stop recording
4. Transcription completes
5. `summarize.py` is invoked automatically on the new transcript
6. Summary saved to `summaries/..._SUM.md`

---

## Step 6: Update README

Add new section under "## Summarization (Optional)":

```markdown
## Summarization

Uses Qwen2.5-7B-Instruct (GGUF Q8_0) via llama.cpp for local CPU inference.

### Setup

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Install summarization dependencies
uv pip install -e ".[summarize]"

# Download model (~6.7 GB)
python download_summarizer.py
```

### Usage

```powershell
# Standalone - summarize a specific file
python summarize.py --qwen transcripts/2026-04-08_14h-32m--5min.md

# With custom context limit
python summarize.py --qwen --max-cont 2000 transcripts/file.md

# Auto-summarize after transcription
python transcribe.py --voxtral --mic --sum
python transcribe.py --parakeet --mic --sum
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--qwen` | required | Model engine selector |
| `--max-cont` | 4000 | Max input tokens (truncation) |
| `--max-out` | 500 | Max output tokens |

### Output

Saved to `summaries/` with format `[transcription_filename]_SUM.md`:

```markdown
# Summary: 2026-04-08_14h-32m--5min

## Model: Qwen2.5-7B-Instruct (GGUF Q8_0, 8-bit)
## Context: 4000 tokens (truncated from input)
## Generated: 2026-04-08 14:45 WITA

---

### SUMMARY

[Abstractive 2-4 sentence summary]

---

### KEY QUOTES

- "[Verbatim quote 1]"
- "[Verbatim quote 2]"
- ...

---

### METRICS
- Input tokens: X
- Output tokens: X
- Processing time: X seconds
```

### Hardware Requirements

- **RAM**: ~12 GB free (8 GB model + 4 GB KV cache)
- **CPU**: Any modern multi-core (uses os.cpu_count() threads)
- **No GPU required**

### Expected Latency

| Phase | Estimated |
|-------|-----------|
| Model load | 30-60s (first run) |
| Input processing (4000 tokens) | 15-30s |
| Generation (500 tokens) | 10-20s |
| **Total** | **~25-60s** |
```

---

## Step 7: Update pyproject.toml

Add summarization dependencies:

```toml
[project.optional-dependencies]
# ... existing ...

# Qwen2.5-7B Summarization (llama.cpp)
summarize = [
    "llama-cpp-python>=0.2.0",
]
```

---

## Estimated Latency (CPU-only on DRAM)

| Phase | Estimated |
|-------|-----------|
| Model load (first run) | 30-60s |
| Input processing (4000 tokens) | 15-30s |
| Generation (500 tokens) | 10-20s |
| **Total** | **~25-60s** |

---

## Implementation Checklist

- [ ] Update pyproject.toml with `llama-cpp-python` dependency
- [ ] Create download_summarizer.py
- [ ] Create summarize.py with full implementation
- [ ] Update transcribe.py to support --sum flag
- [ ] Update README.md with summarization docs
- [ ] Test: Run download_summarizer.py
- [ ] Test: Run summarize.py on a transcript
- [ ] Test: Run transcribe.py --sum workflow

---

## Notes

- **Batch size**: Hardcoded to 1 (as requested)
- **Temperature**: 0.3 (low for deterministic output)
- **Quotes extraction**: Uses regex to find `- "..."` or just `"..."` patterns
- **Input truncation**: Keeps last MAX_CONT tokens (recent content more relevant for summaries)
- **Error handling**: Exits with clear error if model file not found