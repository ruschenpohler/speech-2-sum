# Local Speech-to-Text

Private, offline speech-to-text on CPU. Two model backends are available — pick one based on your hardware and accuracy needs.

| | Voxtral Mini 4B | Parakeet TDT 0.6B |
|---|---|---|
| Source | Mistral AI | NVIDIA |
| Parameters | 4B | 0.6B |
| Weights | ~8 GB (BF16) | ~2.5 GB (FP32) |
| Inference | Streaming (token-by-token) | Batch (full audio at once) |
| CPU speed | ~0.5–2× real-time | Faster on CPU |
| Delay tuning | Yes (`--del`) | No |
| Toolkit | HuggingFace Transformers | NVIDIA NeMo |
| Windows native | Yes | WSL2 recommended |

Both models run fully locally — no audio ever leaves your machine.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.10+.

```powershell
git clone https://github.com/ruschenpohler/speech-2-sum.git
cd speech-2-sum

# Interactive setup — choose Voxtral, Parakeet, or both
python setup.py
```

`setup.py` will create a `.venv`, install the right dependencies for your chosen backend, and print the next steps.

### Manual install (alternative)

```powershell
uv venv .venv
.\.venv\Scripts\Activate.ps1

uv pip install -e ".[voxtral]"   # Mistral Voxtral
uv pip install -e ".[parakeet]"  # NVIDIA Parakeet
uv pip install -e ".[all]"       # both
```

### Download model weights

Run once before first use. Weights are cached for all future transcriptions.

```powershell
python download_voxtral.py          # ~8 GB
python download_parakeet.py         # ~2.5 GB

# Optional: custom cache location
python download_voxtral.py  --cache-dir D:\models
python download_parakeet.py --cache-dir D:\models
```

---

## Quick Start

```powershell
.\.venv\Scripts\Activate.ps1

# Transcribe a file
python transcribe.py --voxtral  audio.mp3
python transcribe.py --parakeet audio.mp3

# Record from microphone
python transcribe.py --voxtral  --mic --dur 30
python transcribe.py --parakeet --mic --dur 30
```

---

## Usage

### All flags

| Flag | Values | Applies to |
|---|---|---|
| `--voxtral` / `--parakeet` | — | `transcribe.py` dispatcher |
| `--mic` | — | both |
| `audio.mp3` | any audio file | both |
| `--dur N` | minutes (default: 60) | both |
| `--del N` | ms: 80, 480, 960, 2400 (default: 960) | Voxtral only |
| `--cache-dir PATH` | directory | both |

`--del` controls Voxtral's transcription delay: higher values improve accuracy at the cost of latency. It is accepted but ignored by Parakeet for CLI consistency.

### Model-specific scripts (also work standalone)

```powershell
python transcribe_voxtral.py  --mic --dur 60 --del 960
python transcribe_parakeet.py --mic --dur 30
```

---

## Output

Transcripts are saved to `transcripts/` as timestamped Markdown files:

```
transcripts/2026-04-08_14h-32m--5min.md
```

Each file header includes the model name, timestamp, approximate location (from IP), WiFi network, and audio source.

---

## Notes

- **Parakeet on Windows:** NVIDIA NeMo is not officially supported on native Windows. If you hit installation issues, use WSL2.
- **Voxtral on CPU:** Expect ~0.5–2× real-time factor depending on CPU.
- **Supported audio formats:** mp3, wav, flac, ogg, and other formats supported by librosa/soundfile.

## Fallback

If Transformers is too slow for Voxtral, [antirez/voxtral.c](https://github.com/antirez/voxtral.c) is a faster pure-C implementation.
