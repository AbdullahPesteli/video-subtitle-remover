# OpenCode Go Integration Guide

This document explains how to use the Subtitle Remover with OpenCode Go (and other gstack-compatible agents).

## Overview

The Subtitle Remover is integrated as a **gstack skill**, making it available to:
- OpenCode Go agents
- Claude Code (with gstack)
- Any other gstack-compatible AI system

## Setup

### 1. Install gstack (if not already installed)

```bash
# Using Bun
bun install -g garrytan/gstack

# Or from source
git clone https://github.com/garrytan/gstack.git
cd gstack && bun install && bun run build
```

### 2. Copy skill to gstack

```bash
cp -r gstack-integration/SKILL.md ~/.gstack/skills/subtitle-remove/
cp gstack-integration/gstack-subtitle-remove ~/.gstack/bin/
chmod +x ~/.gstack/bin/gstack-subtitle-remove
```

### 3. Install SubtitleRemover.app

```bash
cp -r dist/SubtitleRemover.app /Applications/
```

## Usage with OpenCode Go

### Single Video

```bash
# Basic invocation
opencode /subtitle-remove ~/Videos/movie.mp4

# With options
opencode /subtitle-remove ~/Videos/movie.mp4 --model lama --no-gpu
```

## Performance Notes

**Apple Silicon:** Use `--no-gpu` (faster than GPU on M1/M2/M3)

## Support

GitHub: https://github.com/YaoFANGUK/video-subtitle-remover
