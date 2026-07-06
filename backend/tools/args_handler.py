import argparse

from .constant import InpaintMode

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Video Subtitle Remover Command Line Tool"
    )
    parser.add_argument(
        "positional_input", nargs="?", type=str,
        help="Input video file path (same as --input)."
    )
    parser.add_argument(
        "--input", "-i", "--video", required=False, type=str,
        help="Input video file path"
    )
    parser.add_argument(
        "--output", "-o", required=False, type=str, default=None,
        help="Output video file path (optional)"
    )
    parser.add_argument(
        "--subtitle-area-coords", "-c", action="append", nargs=4, type=int, metavar=("YMIN", "YMAX", "XMIN", "XMAX"),
        help="Subtitle area coordinates (ymin ymax xmin xmax). Can be specified multiple times for multiple areas."
    )
    parser.add_argument(
        "--subtitle-area-ratio", action="append", nargs=4, type=float, metavar=("YMIN", "YMAX", "XMIN", "XMAX"),
        help="Subtitle area as frame ratios in 0..1. Can be specified multiple times."
    )
    parser.add_argument(
        "--inpaint-mode", "--model", type=str, default="opencv",
        choices=[mode.name.lower().replace('_','-') for mode in InpaintMode],
        help="Inpaint mode, default is opencv"
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Disable hardware acceleration and force CPU-only mode."
    )
    parser.add_argument(
        "--detect-fps", type=int, default=None,
        help="Subtitle OCR sampling rate in frames per second. Default comes from config."
    )
    parser.add_argument(
        "--ocr-max-dim", type=int, default=None,
        help="Resize OCR input so its longest side is at most this value."
    )
    parser.add_argument(
        "--mask-deviation", type=int, default=None,
        help="Expand detected subtitle boxes by this many pixels before inpainting."
    )
    parser.add_argument(
        "--mask-mode", choices=["box", "line", "area"], default="area",
        help="Mask shape for detected subtitle intervals: box keeps OCR boxes, line expands each box to subtitle-area width, area masks the full selected subtitle area. Default is area."
    )
    parser.add_argument(
        "--allow-slow-models", action="store_true",
        help="Allow models that are measured as impractical on this machine, such as ProPainter on macOS/MPS fallback or LAMA CPU."
    )
    parser.add_argument(
        "--profile-json", type=str, default=None,
        help="Write timing and runtime metadata to this JSON path."
    )
    args = parser.parse_args(argv)
    if args.input is None:
        args.input = args.positional_input
    if args.input is None:
        parser.error("input video path is required; pass it positionally or with --input")
    delattr(args, "positional_input")
    args.inpaint_mode = InpaintMode[args.inpaint_mode.replace('-','_').upper()]
    if args.subtitle_area_coords is None:
        args.subtitle_area_coords = []
    if args.subtitle_area_ratio is None:
        args.subtitle_area_ratio = []
    return args
