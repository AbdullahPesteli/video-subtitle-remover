#!/usr/bin/env python3
"""Headless smoke test for subtitle detection/removal.

Creates a short synthetic video with a subtitle band, runs the backend CLI, and
verifies that OCR no longer detects text in the cleaned output.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from backend.config import config  # noqa: E402
from backend.tools.subtitle_detect import SubtitleDetect  # noqa: E402


def make_synthetic_video(path: Path, fps: int = 24, frames: int = 48) -> tuple[int, int]:
    width, height = 640, 360
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create synthetic video writer: {path}")
    for _ in range(frames):
        frame = np.full((height, width, 3), (32, 34, 38), dtype=np.uint8)
        cv2.rectangle(frame, (80, 292), (560, 336), (245, 245, 245), -1)
        cv2.putText(
            frame,
            "THIS SUBTITLE SHOULD DISAPPEAR",
            (96, 322),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Synthetic video was not created: {path}")
    return width, height


def assert_readable_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    info = {
        "path": str(path),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else 0,
        "readable": bool(ok),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "shape": list(frame.shape) if ok else None,
    }
    cap.release()
    if not info["readable"]:
        raise RuntimeError(f"Video is not readable: {path}")
    return info


def detect_count(path: Path, area: tuple[int, int, int, int]) -> tuple[int, float]:
    config.subtitleDetectionSampleFps.value = 2
    config.subtitleDetectionMaxDimension.value = 1280
    detector = SubtitleDetect(str(path), [area])
    start = time.time()
    result = detector.find_subtitle_frame_no()
    return len(result), time.time() - start


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a headless SubtitleRemover self harness.")
    parser.add_argument("--model", default="opencv", choices=["opencv", "lama", "sttn-auto", "sttn-det", "propainter"])
    parser.add_argument("--work-dir", default="/tmp")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "vsr_self_harness_input.mp4"
    output_path = work_dir / f"vsr_self_harness_{args.model}.mp4"
    output_path.unlink(missing_ok=True)

    width, height = make_synthetic_video(input_path)
    area = (round(height * 0.65), round(height * 0.98), round(width * 0.05), round(width * 0.95))

    before_count, before_seconds = detect_count(input_path, area)
    command = [
        sys.executable,
        "-m",
        "backend.main",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--model",
        args.model,
        "--no-gpu",
        "--detect-fps",
        "2",
        "--ocr-max-dim",
        "1280",
        "--subtitle-area-ratio",
        "0.65",
        "0.98",
        "0.05",
        "0.95",
    ]
    start = time.time()
    completed = subprocess.run(command, cwd=PROJECT_DIR, text=True, capture_output=True)
    process_seconds = time.time() - start
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        return completed.returncode

    output_info = assert_readable_video(output_path)
    after_count, after_seconds = detect_count(output_path, area)
    summary = {
        "status": "success" if before_count > 0 and after_count == 0 else "failed",
        "model": args.model,
        "input": assert_readable_video(input_path),
        "output": output_info,
        "ocr_detected_before": before_count,
        "ocr_detected_after": after_count,
        "timing_seconds": {
            "detect_before": round(before_seconds, 2),
            "process": round(process_seconds, 2),
            "detect_after": round(after_seconds, 2),
        },
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
