# Summarization Pipeline Implementation Plan
# Qwen2.5-7B (llama.cpp, GGUF Q8_0) - CPU Inference on DRAM

## Overview

Add a local text summarization feature that:
- Uses Qwen2.5-7B-Instruct (GGUF Q8_0, 8-bit quantized)
- Runs via llama.cpp on CPU (DRAM only, ~12GB total)
- Produces **hybrid output**: abstractive summary + extractive quotes
- Saves to `summaries/[transcription_filename]_SUM.md`

---

## Design Decisions

**No `--qwen` flag.** There is currently one summarization model, so no selector flag is needed. If a second summarizer is added later, follow the same pattern as transcription: add `--qwen`/`--[newmodel]` as mutually exclusive flags to `summarize.py` at that point, and add a `summarize.py` dispatcher analogous to `transcribe.py`.

**Import shared utilities from `utils.py`.** `get_wifi_name()`, `get_location()`, `print_ram()`, and `save_transcript()` are never duplicated inline.

**Input truncation with `--keep`.** Two strategies currently implemented for handling transcripts that exceed `--max-cont` tokens:
- `last` *(default)* — keep the final N tokens, prepend `[... OPENING OMITTED ...]`. Best for meetings and conversations where conclusions, decisions, and action items tend to come late.
- `first` — keep the opening N tokens, append `[... REMAINDER OMITTED ...]`. Best for lectures, presentations, and structured talks where the thesis and agenda are front-loaded.

**TODO — implement when truncation is first encountered in practice** (i.e. the first time you see `[... OPENING OMITTED ...]` or `[... REMAINDER OMITTED ...]` appear in a summary output and feel the missing section mattered): extend `--keep` with:
- `mid` — anchor the first G and last G words untouched; keep a contiguous middle window of N tokens from the body between them. Good when the opening and closing are both critical (e.g. a client call with a clear intro and wrap-up) and the dense middle can be sampled as one block.
- `mid-sliced` — same G-word anchors, then uniformly sample the middle body in K equal slices with `[... SECTION OMITTED ...]` markers between each. Better than `mid` when important content is spread throughout a long recording (e.g. a multi-topic all-hands). Add `--slices K` (default 3) and `--anchor-words G` (default 100) as companion flags.

**Transcript path passed explicitly to `--sum`.** The dispatcher finds the output path from the transcription subprocess rather than inferring by modification time, avoiding races between concurrent sessions.

**Models live in `s2t/models/`.** Already covered by `.gitignore` via `*.gguf`.

---

## File Structure (Post-Implementation)

```
s2t/
├── .venv/
├── utils.py                        # Shared utilities
├── transcribe.py                   # Dispatcher: --voxtral, --parakeet [--sum]
├── transcribe_voxtral.py
├── transcribe_parakeet.py
├── summarize.py                    # NEW
├── download_summarizer.py          # NEW
├── models/                         # GGUF weights (gitignored)
│   └── qwen2.5-7b-instruct-q8_0.gguf
├── transcripts/
└── summaries/                      # NEW
    └── 2026-04-08_14h-32m--5min_SUM.md
```

---

## Step 1: Dependencies

### pyproject.toml addition

```toml
# Qwen2.5-7B Summarization (llama.cpp)
summarize = [
    "llama-cpp-python>=0.2.0",
]
```

### Install

```powershell
uv pip install -e ".[summarize]"
```

---

## Step 2: Download model (~6.7 GB)

```powershell
python download_summarizer.py
python download_summarizer.py --cache-dir D:\\models   # optional custom path
```

Source: `second-state/Qwen2.5-7B-Instruct-GGUF` on HuggingFace, file `Qwen2.5-7B-Instruct-Q8_0.gguf`.

| Property | Value |
|---|---|
| Model | Qwen2.5-7B-Instruct |
| Format | GGUF (llama.cpp) |
| Quantization | Q8_0 (8-bit) |
| Size | ~6.7 GB |
| RAM usage | ~8 GB weights + ~4 GB KV cache ≈ 12 GB total |

---

## Step 3: CLI interface

```powershell
# Standalone
python summarize.py transcripts/2026-04-08_14h-32m--5min.md

# Truncation strategy
python summarize.py --keep first transcripts/file.md   # keep opening N tokens
python summarize.py --keep last  transcripts/file.md   # keep final N tokens (default)

# Context / output limits
python summarize.py --max-cont 2000 --max-out 300 transcripts/file.md

# Auto-summarize after transcription
python transcribe.py --voxtral  --mic --sum
python transcribe.py --parakeet --mic --sum
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `input_file` | required | Path to transcript `.md` file |
| `--keep` | `last` | Truncation strategy: `first` or `last` (see design decisions for when to add `mid` and `mid-sliced`) |
| `--max-cont` | 4000 | Max input tokens before truncation |
| `--max-out` | 500 | Max output tokens for summary |
| `--cache-dir` | `./models` | Path to directory containing the GGUF file |

---

## Step 4: Output format

Saved to `summaries/[transcript_stem]_SUM.md`:

```markdown
# Summary: 2026-04-08_14h-32m--5min

**Model:** Qwen2.5-7B-Instruct (GGUF Q8_0) | **Keep:** last | **Input tokens:** 1842

Generated: 2026-04-08 14:45:00 | Berlin, Germany | WiFi: MyNetwork

---

### Summary

[2-4 sentence abstractive summary]

---

### Key Quotes

- "[Verbatim quote 1]"
- "[Verbatim quote 2]"

---

### Metrics

- Processing time: 42s | Tokens/s: 11.9
```

---

## Step 5: Integration with transcribe.py

`--sum` flag added to dispatcher. After transcription completes successfully, the transcript path is passed directly to `summarize.py` (no mtime inference).

---

## Hardware requirements

- RAM: ~12 GB free
- CPU: any modern multi-core (uses `os.cpu_count()` threads automatically)
- No GPU required

## Error Handling

| Failure Mode | User Message | Action |
|---|---|---|
| GGUF file missing | `Error: Model not found at <path>. Run: python download_summarizer.py` | Exit with code 1 |
| GGUF file corrupted | `Error: Failed to load GGUF model. File may be corrupted. Re-download: python download_summarizer.py --cache-dir <path>` | Exit with code 1 |
| Input file missing | `Error: File not found: <path>` | Exit with code 1 |
| Input file unreadable | `Error: Cannot read file. Check permissions: <path>` | Exit with code 1 |
| Model load timeout | `[TIMEOUT] Model load exceeded 120s. Your system may be undersized for 12GB model.` | Exit with code 1 |
| Generation timeout | `[TIMEOUT] Generation exceeded 300s. Try reducing --max-out.` | Exit with code 1 |
| Empty transcription | `Warning: Transcription file is empty. Nothing to summarize.` | Skip, no output file |
| Network error (location/WiFi) | `[Warning] Could not fetch location/WiFi. Continuing without metadata.` | Continue, use "Unknown" |

All errors printed to stderr, with clear action items. No silent failures.

---

## Expected latency (CPU-only)

| Phase | Estimate |
|---|---|
| Model load | 30–60s |
| Input processing (4000 tokens) | 15–30s |
| Generation (500 tokens) | 10–20s |
| **Total** | **~55–110s** |

---

## Implementation Checklist

- [x] Design decisions documented
- [ ] pyproject.toml updated (llama-cpp-python dependency)
- [ ] download_summarizer.py created
- [ ] summarize.py created (with clear error messages as specified above)
- [ ] transcribe.py updated with --sum
- [x] README.md updated
- [ ] Tested: download_summarizer.py
- [ ] Tested: summarize.py standalone
- [ ] Tested: transcribe.py --sum workflow
