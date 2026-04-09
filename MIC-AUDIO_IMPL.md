# Microphone + System Audio Recording Implementation Plan

## Overview

Add the ability to record both physical microphone AND system audio (via WASAPI loopback) simultaneously. This enables capturing both your voice AND other meeting participants' voices in a single transcription.

---

## Design Decisions

**Sample rate**: Fixed at 16kHz (`--kHz 16` flag, default 16). Both Parakeet and Voxtral expect 16kHz input, so this is universal.

**Audio mixing**: When both mic and loopback are recorded, each stream is normalized to equal max amplitude before averaging. This ensures your voice and meeting participants' voices contribute equally regardless of original volume differences.

---

## File Structure (Post-Implementation)

```
s2t/
├── utils.py                        # Added record_audio() function
├── transcribe_voxtral.py           # Refactored argparse + transcribe_from_mic()
├── transcribe_parakeet.py          # Refactored argparse + transcribe_from_mic()
├── README.md                       # Updated with --kHz flag
└── MIC-AUDIO_IMPL.md               # This file
```

---

## Step 1: Add `record_audio()` to utils.py

```python
def record_audio(mic: bool, loopback: bool, duration_sec: float, kHz: int = 16) -> tuple[np.ndarray, str]:
    """
    Record audio from mic, system loopback, or both simultaneously.

    Args:
        mic: record physical microphone
        loopback: record system audio via WASAPI loopback
        duration_sec: max recording length in seconds
        kHz: sample rate in kHz (default: 16)

    Returns:
        (audio_array, source_label)
        - audio_array: float32 numpy array, mono, at kHz*1000 Hz
        - source_label: "Microphone", "System Audio (loopback)", or "Microphone + System Audio"
    """
```

**Implementation:**
1. Find loopback device using existing `get_loopback_device()`
2. Open 1 or 2 `sd.InputStream` concurrently using a context manager
3. Record into separate buffers
4. If both: normalize each to max amplitude, then average (handles clock drift by trimming to shorter duration)
5. Return audio array + source label

---

## Step 2: Fix argparse in transcribe_voxtral.py and transcribe_parakeet.py

**Current structure (broken):**
```python
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("audio_file", ...)
group.add_argument("--mic", ...)
group.add_argument("--mic-loop", ...)
```

**New structure:**
```python
# Recording mode: at least one of --mic or --mic-loop must be set
recording_group = parser.add_argument_group(title="recording options")
recording_group.add_argument("--mic", action="store_true", help="Record from microphone")
recording_group.add_argument("--mic-loop", action="store_true", help="Record system audio via WASAPI loopback")
recording_group.add_argument("--kHz", type=int, default=16, help="Sample rate in kHz (default: 16)")

# File input: mutually exclusive with recording
file_group = parser.add_argument_group(title="file input")
file_group.add_argument("audio_file", nargs="?", help="Path to audio file (mp3, wav, flac, etc.)")

# Validation in main():
if args.audio_file and (args.mic or args.mic_loop):
    parser.error("Cannot use audio_file with --mic or --mic-loop")
if not args.audio_file and not args.mic and not args.mic_loop:
    parser.error("Must specify either audio_file or at least one of --mic/--mic-loop")
```

**Valid combinations:**
- `python transcribe.py audio.mp3` — file input
- `python transcribe.py --mic` — mic only
- `python transcribe.py --mic-loop` — system audio only
- `python transcribe.py --mic --mic-loop` — both, mixed

---

## Step 3: Refactor `transcribe_from_mic()` in both scripts

Simplified signature:
```python
def transcribe_from_mic(model, sample_rate, duration_sec, mic, loopback):
    audio_array, source_label = record_audio(
        mic=mic,
        loopback=loopback,
        duration_sec=duration_sec,
        kHz=sample_rate // 1000  # convert Hz to kHz
    )
    # ... rest of transcription
```

---

## Step 4: Update imports

```python
# Before:
from utils import print_ram, save_transcript, record_until_esc_or_timeout, get_loopback_device

# After:
from utils import print_ram, save_transcript, record_audio
```

---

## Step 5: Update README.md

Add `--kHz` flag to the flags table:

| Flag | Values | Applies to |
|---|---|---|
| `--mic` | — | both |
| `--mic-loop` | — | both |
| `--kHz` | integer (default: 16) | both |

---

## Implementation Checklist

- [ ] Add `record_audio()` to utils.py
- [ ] Fix argparse in transcribe_voxtral.py (separate file/recording groups)
- [ ] Fix argparse in transcribe_parakeet.py (separate file/recording groups)
- [ ] Refactor transcribe_from_mic() in transcribe_voxtral.py to use record_audio()
- [ ] Refactor transcribe_from_mic() in transcribe_parakeet.py to use record_audio()
- [ ] Update imports in both scripts
- [ ] Update README.md with --kHz flag
- [ ] Test: --mic only
- [ ] Test: --mic-loop only
- [ ] Test: --mic --mic-loop (both)
- [ ] Test: audio_file still works
- [ ] Test: error on audio_file + --mic