import os
import logging
import site
import sys
import warnings

# Keep third-party caches outside the read-only app bundle. This also avoids
# repeated "building font cache" noise in frozen CLI runs.
_app_cache_dir = os.path.expanduser("~/Library/Caches/SubtitleRemover")
try:
    os.makedirs(_app_cache_dir, exist_ok=True)
except Exception:
    pass
for _name in ("matplotlib", "paddle", "paddlex"):
    _path = os.path.join(_app_cache_dir, _name)
    try:
        os.makedirs(_path, exist_ok=True)
    except Exception:
        pass
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_app_cache_dir, "matplotlib"))
os.environ.setdefault("PADDLE_HOME", os.path.join(_app_cache_dir, "paddle"))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", os.path.join(_app_cache_dir, "paddlex"))

# Allow PyTorch to keep using MPS while falling back to CPU for unsupported ops
# such as torchvision::deform_conv2d used by ProPainter.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


_NOISY_STDERR_PATTERNS = (
    "Matplotlib is building the font cache",
    "No ccache found. Please be aware that recompiling all source files may be required.",
    "Connectivity check to the model hoster has been skipped because `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` is enabled.",
)


def _is_known_third_party_noise(message):
    return any(pattern in message for pattern in _NOISY_STDERR_PATTERNS)


class _FilteredStderr:
    def __init__(self, stream):
        self._stream = stream
        self._buffer = ""
        self._vsr_noise_filter = True

    def write(self, text):
        if not isinstance(text, str):
            return self._stream.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            output = line + "\n"
            if not _is_known_third_party_noise(line):
                self._stream.write(output)
        return len(text)

    def flush(self):
        if self._buffer:
            if not _is_known_third_party_noise(self._buffer):
                self._stream.write(self._buffer)
            self._buffer = ""
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _ThirdPartyNoiseFilter(logging.Filter):
    def filter(self, record):
        return not _is_known_third_party_noise(record.getMessage())


def _install_runtime_noise_filters():
    if not getattr(sys.stderr, "_vsr_noise_filter", False):
        sys.stderr = _FilteredStderr(sys.stderr)

    warnings.filterwarnings(
        "ignore",
        message=r"No ccache found\. Please be aware that recompiling all source files may be required\..*",
        category=UserWarning,
    )

    paddle_logger = logging.getLogger("paddlex")
    if not any(isinstance(filter_, _ThirdPartyNoiseFilter) for filter_ in paddle_logger.filters):
        paddle_logger.addFilter(_ThirdPartyNoiseFilter())


_install_runtime_noise_filters()


def _patch_pyinstaller_site_packages():
    if not getattr(sys, "frozen", False):
        return

    original_getsitepackages = getattr(site, "getsitepackages", None)

    def getsitepackages():
        candidates = []
        if original_getsitepackages is not None:
            try:
                candidates.extend(original_getsitepackages() or [])
            except Exception:
                pass

        bundle_root = getattr(sys, "_MEIPASS", None)
        executable_dir = os.path.dirname(sys.executable)
        candidates.extend([
            bundle_root,
            executable_dir,
            os.path.abspath(os.path.join(executable_dir, "..", "Frameworks")),
            os.path.abspath(os.path.join(executable_dir, "..", "Resources")),
        ])

        paths = []
        for candidate in candidates:
            if not candidate:
                continue
            path = os.fspath(candidate)
            if path not in paths:
                paths.append(path)
        return paths

    site.getsitepackages = getsitepackages


_patch_pyinstaller_site_packages()

# 忽略所有的 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
