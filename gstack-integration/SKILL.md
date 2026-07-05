---
name: subtitle-remove
version: 1.0.0
description: Remove hardcoded subtitles from videos using AI inpainting (macOS, gstack)
triggers:
  - remove subtitles
  - strip subtitles
  - subtitle remover
  - ai video inpainting
allowed-tools:
  - Bash
  - Read
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "bash $HOME/.gstack/bin/subtitle-remove-check.sh"
          statusMessage: "Validating video and model selection..."
---

# /subtitle-remove — AI-Powered Video Subtitle Removal

Remove hardcoded/burned-in subtitles from videos while preserving original resolution.
Uses state-of-the-art inpainting models (STTN, LAMA, ProPainter) with automatic subtitle detection.

## When to invoke this skill

Use when you need to:
- Remove burned-in subtitles from videos
- Clean up watermarked or overlaid text
- Process batch video files
- Support multiple languages (auto-detection)

Invoke with: `/subtitle-remove <video-path> [options]`

## Basic usage

```bash
# Single video (auto-detect subtitle language, use LAMA model)
/subtitle-remove ~/Videos/movie.mp4 --output ~/Videos/movie_clean.mp4

# Specify model (STTN=fast, LAMA=good-for-animation, ProPainter=best-quality)
/subtitle-remove ~/Videos/movie.mp4 --model lama

# Batch process directory
/subtitle-remove ~/Videos/ --batch --model lama --max-parallel 2

# Disable GPU (CPU-only, faster on Apple Silicon M1/M2/M3)
/subtitle-remove ~/Videos/movie.mp4 --no-gpu
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output` / `-o` | `{input}_cleaned.{ext}` | Output video path |
| `--model` | `lama` | Inpainting model: `sttn`, `lama`, `propainter` |
| `--language` | `auto` | Subtitle language for detection: `auto`, `en`, `zh`, `es`, `fr`, `ja`, `ko` |
| `--batch` | `false` | Process all .mp4/.mkv in directory |
| `--max-parallel` | `2` | Parallel jobs (CPU-bound, keep ≤ 2-4) |
| `--no-gpu` | `false` | Force CPU-only (faster on Apple Silicon) |
| `--quality` | `medium` | Detection confidence: `low`, `medium`, `high` |
| `--format` | `json` | Output format: `json`, `csv` |

## Performance & Requirements

**Hardware:**
- Apple Silicon (M1/M2/M3): 2-5 min per 10-min video (CPU-only)
- Intel Mac: 3-10 min per 10-min video (CPU-only)
- NVIDIA GPU: 1-2 min per 10-min video (CUDA, if available)

**Models:**
- **STTN** (FAST): 3-5 min/10min video, good quality
- **LAMA** (RECOMMENDED): 2-3 min/10min video, best for animation
- **ProPainter** (BEST): 10-20 min/10min video, highest quality

**Storage:** ~3.4GB for bundled app (PyInstaller + PyTorch + models)

## Configuration

User preferences stored in `~/.gstack/subtitle-remove.json`:

```json
{
  "model": "lama",
  "auto_detect_language": true,
  "disable_gpu": false,
  "batch_max_parallel": 2,
  "output_format": "json"
}
```

Edit or set via: `gstack-config write subtitle-remove.model sttn`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Processing stuck at %1" | Normal: OCR init (2-3 min) + inference | First run downloads models |
| "Too slow on M1/M2" | Use `--no-gpu` (already CPU-only, fastest) |
| "No subtitles detected" | Try `--quality high` or manual region selection in app |
| "Out of memory" | Reduce batch size: `--max-parallel 1` |
| "App won't open" | One-time Gatekeeper warning: right-click → Open |

## Related skills

- `/browse` — inspect video with Claude Code
- `/qa` — test video output quality
- `/ship` — deploy batch-processed videos

## Supported formats

**Input:** `.mp4`, `.mkv`, `.mov`, `.avi`, `.flv`, `.webm` (OpenCV-compatible)
**Output:** `.mp4` (H.264, original resolution)

## Architecture

**Local execution:** `SubtitleRemover.app` → PyTorch inference (CPU-only)
**No cloud:** all processing on user's machine
**Return:** cleaned video + JSON report (confidence scores, removed regions)

---

**Note:** First run downloads AI models (~500MB, one-time). Subsequent runs use cached models.

