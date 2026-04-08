"""
Qwen2.5-7B-Instruct Summarization — Local CPU Inference
Uses llama.cpp (GGUF Q8_0) for efficient CPU-only inference.
Produces a hybrid output: abstractive summary + extractive key quotes.

Usage:
    python summarize.py transcripts/2026-04-08_14h-32m--5min.md
    python summarize.py --keep last  transcripts/file.md   # default
    python summarize.py --keep first transcripts/file.md
    python summarize.py --max-cont 2000 --max-out 300 transcripts/file.md
"""

import argparse
import datetime
import os
import re
import sys
import time
from pathlib import Path

from llama_cpp import Llama

from utils import print_ram, get_wifi_name, get_location


MODEL_FILE = "qwen2.5-7b-instruct-q8_0.gguf"
DEFAULT_MODEL_DIR = "models"
TEMPERATURE = 0.3
CHARS_PER_TOKEN = 4  # rough estimate for English


SUMMARIZE_PROMPT = """\
You are a professional summarization assistant. Analyze the following transcription and provide:

1. SUMMARY: A concise abstractive summary (2-4 sentences) capturing the key points.
2. KEY QUOTES: 3-5 verbatim extracts from the text that best represent the content.
   Each quote must be a direct extract enclosed in quotation marks.

Format your response exactly as:

SUMMARY:
[Your 2-4 sentence summary here]

KEY QUOTES:
- "[First exact quote from text]"
- "[Second exact quote]"
- "[Third exact quote]"

TRANSCRIPTION:
{input_text}"""


# =============================================================================
# Model loading
# =============================================================================

def load_model(model_path: str):
    """Load Qwen2.5-7B GGUF model via llama.cpp on CPU."""
    n_threads = os.cpu_count() or 4
    print_ram("Before loading")
    print(f"\n[Step 1/2] Loading model from {model_path}...")
    print(f"  Threads: {n_threads} | Device: CPU")

    t0 = time.time()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=0,
        n_threads=n_threads,
        n_ctx=4096,
        verbose=False,
    )
    elapsed = time.time() - t0

    print(f"[Step 1/2] Model loaded in {elapsed:.1f}s")
    print_ram("After loading")
    return llm


# =============================================================================
# Input truncation
# =============================================================================

def truncate(text: str, max_tokens: int, keep: str) -> tuple[str, int]:
    """
    Truncate text to max_tokens using the chosen strategy.
    Returns (truncated_text, estimated_token_count).

    keep='first' : keep the opening N tokens; appends an omission marker
    keep='last'  : keep the final N tokens; prepends an omission marker

    Future strategies (add when truncation is first encountered in practice):
    keep='mid'        : anchor first G and last G words, keep middle N tokens
    keep='mid-sliced' : anchor first G and last G words, uniformly sample
                        the middle in equal slices
    See SUMMARIZE_IMPL.md for the full specification.
    """
    estimated = len(text) // CHARS_PER_TOKEN
    if estimated <= max_tokens:
        return text, estimated

    chars = max_tokens * CHARS_PER_TOKEN
    print(f"  Input ~{estimated} tokens, truncated to ~{max_tokens} (strategy: {keep})")

    if keep == "first":
        truncated = text[:chars] + "\n\n[... REMAINDER OMITTED ...]"
    else:  # last
        truncated = "[... OPENING OMITTED ...]\n\n" + text[-chars:]

    return truncated, max_tokens


# =============================================================================
# Read and parse transcript
# =============================================================================

def read_transcript(input_file: str) -> tuple[str, str]:
    """
    Read a transcript .md file.
    Returns (header_line, body_text) — header is the first # line,
    body is everything after the blank line that follows it.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    header = ""
    body_lines = []
    past_header = False

    for line in lines:
        if not past_header and line.startswith("#"):
            header = line
            continue
        if not past_header and line.strip() == "":
            past_header = True
            continue
        if past_header:
            body_lines.append(line)

    return header, "\n".join(body_lines).strip()


# =============================================================================
# Summarize
# =============================================================================

def summarize(llm, text: str, max_tokens: int, max_out: int) -> tuple[str, float]:
    """Run inference. Returns (raw_output, elapsed_seconds)."""
    prompt = SUMMARIZE_PROMPT.format(input_text=text)

    print(f"\n[Step 2/2] Generating summary...")
    print(f"  Input: ~{len(text) // CHARS_PER_TOKEN} tokens | Max output: {max_out} tokens")

    t0 = time.time()
    output = llm(
        prompt,
        max_tokens=max_out,
        temperature=TEMPERATURE,
        stop=["TRANSCRIPTION:", "\n\n\n"],
    )
    elapsed = time.time() - t0

    raw = output["choices"][0]["text"].strip()
    out_tokens = len(raw) // CHARS_PER_TOKEN
    print(f"  Done in {elapsed:.1f}s (~{out_tokens} tokens, {out_tokens / elapsed:.1f} t/s)")
    return raw, elapsed


# =============================================================================
# Parse output
# =============================================================================

def parse_output(raw: str) -> tuple[str, list[str]]:
    """Extract summary paragraph and quotes list from raw LLM output."""
    summary_match = re.search(r"SUMMARY:\s*(.+?)(?=KEY QUOTES:|$)", raw, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else raw.strip()

    quotes_match = re.search(r"KEY QUOTES:\s*(.+)$", raw, re.DOTALL)
    quotes_text = quotes_match.group(1).strip() if quotes_match else ""

    quotes = re.findall(r'"([^"]+)"', quotes_text)
    if not quotes:
        quotes = [line.lstrip("- ").strip() for line in quotes_text.splitlines() if line.strip()]

    return summary, quotes


# =============================================================================
# Save output
# =============================================================================

def save_summary(
    summary: str,
    quotes: list[str],
    input_file: str,
    input_tokens: int,
    elapsed: float,
    keep: str,
) -> Path:
    """Save summary to summaries/[transcript_stem]_SUM.md."""
    input_path = Path(input_file)
    out_dir = Path("summaries")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / (input_path.stem + "_SUM.md")

    now = datetime.datetime.now()
    wifi = get_wifi_name()
    location = get_location()
    out_tokens = sum(len(q) for q in quotes) // CHARS_PER_TOKEN  # rough

    quotes_block = "\n".join(f'- "{q}"' for q in quotes) if quotes else "_No quotes extracted._"

    content = (
        f"# Summary: {input_path.stem}\n\n"
        f"**Model:** Qwen2.5-7B-Instruct (GGUF Q8_0) | "
        f"**Keep:** {keep} | "
        f"**Input tokens:** {input_tokens}\n\n"
        f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')} | {location} | WiFi: {wifi}\n\n"
        f"---\n\n"
        f"### Summary\n\n"
        f"{summary}\n\n"
        f"---\n\n"
        f"### Key Quotes\n\n"
        f"{quotes_block}\n\n"
        f"---\n\n"
        f"### Metrics\n\n"
        f"- Processing time: {elapsed:.1f}s | "
        f"Tokens/s: {out_tokens / elapsed:.1f}\n"
        f"- Source transcript: `{input_path.name}`\n"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nSaved to: {out_path}")
    return out_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-7B Summarization — Local CPU Inference"
    )
    parser.add_argument(
        "input_file",
        help="Path to transcript .md file to summarize",
    )
    parser.add_argument(
        "--keep",
        choices=["first", "last"],
        default="last",
        help="Truncation strategy when input exceeds --max-cont: "
             "'first' keeps the opening tokens (good for lectures/presentations), "
             "'last' keeps the final tokens (good for meetings where conclusions matter). "
             "Default: last",
    )
    parser.add_argument(
        "--max-cont",
        type=int,
        default=4000,
        help="Max input tokens before truncation (default: 4000)",
    )
    parser.add_argument(
        "--max-out",
        type=int,
        default=500,
        help="Max output tokens for summary (default: 500)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=f"Directory containing the GGUF model file (default: ./{DEFAULT_MODEL_DIR})",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    model_dir = Path(args.cache_dir) if args.cache_dir else Path(DEFAULT_MODEL_DIR)
    model_path = model_dir / MODEL_FILE
    if not model_path.exists():
        print(f"Error: model not found at {model_path}")
        print("Run: python download_summarizer.py")
        sys.exit(1)

    llm = load_model(str(model_path))

    _, body = read_transcript(str(input_path))
    body, input_tokens = truncate(body, args.max_cont, args.keep)

    raw, elapsed = summarize(llm, body, args.max_cont, args.max_out)
    summary, quotes = parse_output(raw)

    print(f"\n{'=' * 60}")
    print("SUMMARY:")
    print(f"{'=' * 60}")
    print(summary)
    if quotes:
        print(f"\n{'=' * 60}")
        print("KEY QUOTES:")
        print(f"{'=' * 60}")
        for q in quotes:
            print(f'- "{q}"')
    print(f"{'=' * 60}")

    save_summary(summary, quotes, str(input_path), input_tokens, elapsed, args.keep)


if __name__ == "__main__":
    main()
