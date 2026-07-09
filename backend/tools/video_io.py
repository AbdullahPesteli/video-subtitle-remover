import os
import queue
import subprocess
import threading
import time

import cv2
import numpy as np

from .ffmpeg_cli import FFmpegCLI


class FramePrefetcher:
    """
    后台线程预解码视频帧，使 I/O 与模型推理重叠。
    接口兼容 cv2.VideoCapture（read/release）。
    """

    def __init__(self, video_cap, buffer_size=10):
        self.cap = video_cap
        self._buffer = queue.Queue(maxsize=buffer_size)
        self._stopped = False
        self._error = None
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        try:
            while not self._stopped:
                ret, frame = self.cap.read()
                self._buffer.put((ret, frame))
                if not ret:
                    break
        except BaseException as exc:
            self._error = exc
            self._buffer.put((False, None))

    def read(self):
        """读取下一帧，接口与 cv2.VideoCapture.read() 一致。"""
        ret, frame = self._buffer.get()
        if self._error is not None and not ret and frame is None:
            raise RuntimeError("Frame prefetcher failed") from self._error
        return ret, frame

    def get(self, propId):
        return self.cap.get(propId)

    def stop(self):
        """停止预读取，不释放底层 video_cap。"""
        self._stopped = True
        try:
            while not self._buffer.empty():
                self._buffer.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=5)

    def release(self):
        self.stop()
        self.cap.release()


class FFmpegVideoWriter:
    """
    通过 FFmpeg 管道写入帧，使用 libx264 编码。
    接口兼容 cv2.VideoWriter（write/release）。
    """

    def __init__(self, output_path, fps, size):
        self.output_path = output_path
        self.frames_written = 0
        self.write_seconds = 0.0
        self.release_seconds = 0.0
        self._closed = False
        w, h = size
        cmd = [
            FFmpegCLI.instance().ffmpeg_path,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            '-preset', 'fast',
            '-loglevel', 'error',
            output_path
        ]
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame):
        """写入一帧（numpy BGR 数组）。"""
        if self._closed:
            raise RuntimeError("Cannot write to FFmpegVideoWriter after release().")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        start = time.perf_counter()
        try:
            self._process.stdin.write(frame.tobytes())
            self.frames_written += 1
        except BrokenPipeError as exc:
            stderr = self._read_stderr()
            raise RuntimeError(f"FFmpeg encoder pipe closed while writing {self.output_path}: {stderr}") from exc
        finally:
            self.write_seconds += time.perf_counter() - start

    def release(self):
        """关闭管道并等待编码完成。"""
        if self._closed:
            return
        start = time.perf_counter()
        self._closed = True
        try:
            if self._process.stdin:
                self._process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self._process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired as exc:
                self._process.kill()
                self._process.wait(timeout=5)
                raise RuntimeError(f"FFmpeg encoder timed out while writing {self.output_path}") from exc
        finally:
            self.release_seconds += time.perf_counter() - start
        if self._process.returncode != 0:
            stderr = self._read_stderr()
            raise RuntimeError(
                f"FFmpeg encoder failed with exit code {self._process.returncode} for {self.output_path}: {stderr}"
            )

    def _read_stderr(self):
        if not self._process.stderr:
            return ""
        try:
            data = self._process.stderr.read()
        except Exception:
            return ""
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace").strip()
        return str(data).strip()
