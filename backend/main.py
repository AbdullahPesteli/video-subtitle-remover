import json
import os
import sys

if __name__ == '__main__' and any(arg.split("=", 1)[0] in ("-h", "--help") for arg in sys.argv[1:]):
    from backend.tools.args_handler import parse_args
    parse_args(sys.argv[1:])
    sys.exit(0)

if __name__ == '__main__':
    from backend.tools.args_handler import parse_args
    from backend.tools.constant import InpaintMode

    _early_args = parse_args(sys.argv[1:])
    _early_error = None
    if not _early_args.allow_slow_models and sys.platform == "darwin" and _early_args.inpaint_mode == InpaintMode.PROPAINTER:
        _early_error = (
            "ProPainter is disabled by default on macOS because this machine measured "
            "about 159s for 48 synthetic frames and timed out at 900s on the real test. "
            "Use OpenCV, or rerun with --allow-slow-models for an explicit experiment."
        )
    elif not _early_args.allow_slow_models and _early_args.inpaint_mode == InpaintMode.LAMA and _early_args.no_gpu:
        _early_error = (
            "LAMA CPU is disabled by default because it timed out at 600s on the real test. "
            "Enable GPU/MPS acceleration or rerun with --allow-slow-models for an explicit experiment."
        )
    if _early_error:
        if _early_args.profile_json:
            output_dir = os.path.dirname(os.path.abspath(_early_args.profile_json))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(_early_args.profile_json, "w", encoding="utf-8") as f:
                json.dump({
                    "status": "failed",
                    "metadata": {
                        "input": os.path.abspath(_early_args.input),
                        "output": os.path.abspath(_early_args.output) if _early_args.output else None,
                        "model": _early_args.inpaint_mode.value,
                        "error": _early_error,
                        "early_guard": True,
                    },
                    "timing_seconds": {},
                    "events": [{"event": "slow_model_rejected", "model": _early_args.inpaint_mode.value}],
                }, f, indent=2)
        print(_early_error, file=sys.stderr)
        sys.exit(1)

import gc
import shutil
import traceback
import subprocess
from pathlib import Path
import threading
import cv2
from contextlib import contextmanager
from functools import cached_property

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.config import *
from backend.tools.hardware_accelerator import HardwareAccelerator
from backend.tools.common_tools import is_video_or_image, is_image_file, get_readable_path, read_image
from backend.inpaint.opencv_inpaint import OpenCVInpaint
from backend.tools.inpaint_tools import create_mask, batch_generator, expand_frame_ranges
from backend.tools.model_config import ModelConfig
from backend.tools.ffmpeg_cli import FFmpegCLI
from backend.tools.subtitle_detect import SubtitleDetect, paddle_hpi_available
from backend.tools.video_io import FramePrefetcher, FFmpegVideoWriter
import tempfile
import multiprocessing
import time
from tqdm import tqdm
import numpy as np


def cpu_torch_device():
    import torch
    return torch.device("cpu")


class RunProfile:
    def __init__(self):
        self.started_at = time.time()
        self.status = "running"
        self.timing_seconds = {}
        self.metadata = {}
        self.events = []

    def set(self, key, value):
        self.metadata[key] = value

    def add_event(self, event, **data):
        payload = {"event": event, "time": round(time.time(), 3)}
        payload.update(data)
        self.events.append(payload)

    def add_time(self, key, seconds):
        self.timing_seconds[key] = round(self.timing_seconds.get(key, 0.0) + float(seconds), 4)

    @contextmanager
    def measure(self, key):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add_time(key, time.perf_counter() - start)

    def to_dict(self):
        return {
            "status": self.status,
            "metadata": self.metadata,
            "timing_seconds": self.timing_seconds,
            "events": self.events,
        }

    def write(self, path):
        output_dir = os.path.dirname(os.path.abspath(path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class SubtitleRemover:
    def __init__(self, vd_path, gui_mode=False):
        self.profile = RunProfile()
        self.profile_json_path = None
        self._run_start_time = None
        self._profile_finalized = False
        self.audio_merge_status = "not_started"
        self.allow_slow_models = False
        self.mask_mode = config.subtitleMaskMode.value.value
        # 线程锁
        self.lock = threading.RLock()
        # 用户指定的字幕区域位置
        self.sub_areas = []
        # 是否为gui运行，gui运行需要显示预览
        self.gui_mode = gui_mode
        self.hardware_accelerator = HardwareAccelerator.instance(enabled=config.hardwareAcceleration.value)
        # 是否使用硬件加速
        self.hardware_accelerator.set_enabled(config.hardwareAcceleration.value)
        self.model_config = ModelConfig()
        # 判断是否为图片
        self.is_picture = is_image_file(str(vd_path))
        # 视频路径
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(get_readable_path(vd_path))
        # 通过视频路径获取视频名称
        self.vd_name = Path(self.video_path).stem
        # 视频帧总数
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        # 视频帧率
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # 视频尺寸
        self.size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.mask_size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.profile.set("input", os.path.abspath(str(vd_path)))
        self.profile.set("frame_count", self.frame_count)
        self.profile.set("fps", self.fps)
        self.profile.set("size", {"width": self.frame_width, "height": self.frame_height})
        # 创建视频临时对象，windows下delete=True会有permission denied的报错
        self.video_temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        # 创建视频写对象（使用 FFmpeg libx264 编码，比 mp4v 质量更好、文件更小）
        try:
            self.video_writer = FFmpegVideoWriter(get_readable_path(self.video_temp_file.name), self.fps, self.size)
            self.profile.set("video_writer", "ffmpeg-libx264")
        except Exception as exc:
            self.profile.add_event("video_writer_fallback", error=str(exc))
            self.video_writer = cv2.VideoWriter(get_readable_path(self.video_temp_file.name), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
            self.profile.set("video_writer", "opencv-mp4v")
        self.video_out_path = os.path.abspath(os.path.join(os.path.dirname(self.video_path), f'{self.vd_name}_no_sub.mp4'))
        self.propainter_inpaint = None
        self.ext = os.path.splitext(vd_path)[-1]
        if self.is_picture:
            pic_dir = os.path.join(os.path.dirname(self.video_path), 'no_sub')
            if not os.path.exists(pic_dir):
                os.makedirs(pic_dir)
            self.video_out_path = os.path.join(pic_dir, f'{self.vd_name}{self.ext}')

        # 总处理进度
        self.progress_total = 0
        self.progress_remover = 0
        self.isFinished = False
        # 是否将原音频嵌入到去除字幕后的视频
        self.is_successful_merged = False
        # 进度监听器列表
        self.progress_listeners = []
        # inpaint的frame_no区域列表, 默认为inpaint所有帧
        self.ab_sections = None

    @staticmethod
    def is_current_frame_no_start(frame_no, continuous_frame_no_list):
        """
        判断给定的帧号是否为开头，是的话返回结束帧号，不是的话返回-1
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no == frame_no:
                return True
        return False

    @staticmethod
    def find_frame_no_end(frame_no, continuous_frame_no_list):
        """
        判断给定的帧号是否为开头，是的话返回结束帧号，不是的话返回-1
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no <= frame_no <= end_no:
                return end_no
        return -1

    def update_progress(self, tbar, increment):
        tbar.update(increment)
        current_percentage = (tbar.n / tbar.total) * 100
        self.progress_remover = int(current_percentage)
        self.progress_total = self.progress_remover
        self.notify_progress_listeners()

    def append_output(self, *args):
        """输出信息到控制台
        Args:
            *args: 要输出的内容，多个参数将用空格连接
        """
        print(*args)
    
    def add_progress_listener(self, listener):
        """
        添加进度监听器
        
        Args:
            listener: 一个回调函数，接收参数 (progress_total, isFinished)
        """
        if listener not in self.progress_listeners:
            self.progress_listeners.append(listener)
    
    def remove_progress_listener(self, listener):
        """
        移除进度监听器
        
        Args:
            listener: 要移除的监听器函数
        """
        if listener in self.progress_listeners:
            self.progress_listeners.remove(listener)
            
    def notify_progress_listeners(self):
        """
        通知所有进度监听器当前进度
        """
        for listener in self.progress_listeners:
            try:
                listener(self.progress_total, self.isFinished)
            except Exception as e:
                traceback.print_exc()

    def update_preview_with_comp(self, frame_ori, frame_comp):
        """
        更新预览
        """
        pass

    def propainter_mode(self, tbar):
        sub_detector = SubtitleDetect(self.video_path, self.sub_areas)
        with self.profile.measure("ocr_detect"):
            sub_list = sub_detector.find_subtitle_frame_no(sub_remover=self)
        self.profile.set("subtitle_detected_frames", len(sub_list))
        if len(sub_list) == 0:
            raise Exception(tr['Main']['NoSubtitleDetected'].format(self.video_path))
        with self.profile.measure("interval_processing"):
            continuous_frame_no_list = sub_detector.find_continuous_ranges_with_same_mask(sub_list)
            scene_div_points = sub_detector.get_scene_div_frame_no(self.video_path)
            continuous_frame_no_list = sub_detector.split_range_by_scene(continuous_frame_no_list,
                                                                              scene_div_points)
        self.profile.set("subtitle_intervals", continuous_frame_no_list)
        del sub_detector
        gc.collect()        
        device = self.hardware_accelerator.device if self.hardware_accelerator.has_cuda() or self.hardware_accelerator.has_mps() else cpu_torch_device()
        self.model_config.ensure_propainter_model()
        from backend.inpaint.propainter_inpaint import PropainterInpaint
        with self.profile.measure("model_load"):
            propainter_inpaint = PropainterInpaint(device, self.model_config.PROPAINTER_MODEL_DIR, config.propainterMaxLoadNum.value)
        self.append_output(tr['Main']['ProcessingStartRemovingSubtitles'])
        index = 0
        # 使用帧预读取，I/O 与推理重叠
        reader = FramePrefetcher(self.video_cap)
        while True:
            ret, frame = reader.read()
            if not ret:
                break
            index += 1
            # 如果当前帧没有水印/文本则直接写
            if index not in sub_list.keys():
                self.video_writer.write(frame)
                # self.append_output(f'write frame: {index}')
                self.update_progress(tbar, increment=1)
                self.update_preview_with_comp(frame, frame)
                continue
            # 如果有水印，判断该帧是不是开头帧
            else:
                # 如果是开头帧，则批推理到尾帧
                if self.is_current_frame_no_start(index, continuous_frame_no_list):
                    # self.append_output(f'No 1 Current index: {index}')
                    start_frame_no = index
                    # self.append_output(f'find start: {start_frame_no}')
                    # 找到结束帧
                    end_frame_no = self.find_frame_no_end(index, continuous_frame_no_list)
                    # 判断当前帧号是不是字幕起始位置
                    # 如果获取的结束帧号不为-1则说明
                    if end_frame_no != -1:
                        # self.append_output(f'find end: {end_frame_no}')
                        # ************ 读取该区间所有帧 start ************
                        temp_frames = list()
                        # 将头帧加入处理列表
                        temp_frames.append(frame)
                        inner_index = 0
                        # 一直读取到尾帧
                        while index < end_frame_no:
                            ret, frame = reader.read()
                            if not ret:
                                break
                            index += 1
                            temp_frames.append(frame)
                        # ************ 读取该区间所有帧 end ************
                        if len(temp_frames) < 1:
                            # 没有待处理，直接跳过
                            continue
                        elif len(temp_frames) == 1:
                            inner_index += 1
                            single_mask = create_mask(self.mask_size, sub_list[index])
                            with self.profile.measure("model_inpaint"):
                                inpainted_frame = self.lama_inpaint.inpaint(frame, single_mask)
                            self.video_writer.write(inpainted_frame)
                            # self.append_output(f'write frame: {start_frame_no + inner_index} with mask {sub_list[start_frame_no]}')
                            self.update_progress(tbar, increment=1)
                            continue
                        else:
                            # 将读取的视频帧分批处理
                            # 1. 获取当前批次使用的mask
                            mask = create_mask(self.mask_size, sub_list[start_frame_no])
                            for batch in batch_generator(temp_frames, config.propainterMaxLoadNum.value):
                                # 2. 调用批推理
                                if len(batch) == 1:
                                    single_mask = create_mask(self.mask_size, sub_list[start_frame_no])
                                    with self.profile.measure("model_inpaint"):
                                        inpainted_frame = self.lama_inpaint.inpaint(frame, single_mask)
                                    self.video_writer.write(inpainted_frame)
                                    # self.append_output(f'write frame: {start_frame_no + inner_index} with mask {sub_list[start_frame_no]}')
                                    inner_index += 1
                                    self.update_progress(tbar, increment=1)
                                elif len(batch) > 1:
                                    with self.profile.measure("model_inpaint"):
                                        inpainted_frames = propainter_inpaint(batch, mask)
                                    for i, inpainted_frame in enumerate(inpainted_frames):
                                        self.video_writer.write(inpainted_frame)
                                        # self.append_output(f'write frame: {start_frame_no + inner_index} with mask {sub_list[index]}')
                                        inner_index += 1
                                        self.update_preview_with_comp(np.clip(batch[i]+mask[:,:,np.newaxis]*0.3,0,255).astype(np.uint8), inpainted_frame)
                                self.update_progress(tbar, increment=len(batch))

    def sttn_auto_mode(self, tbar):
        """
        使用sttn对选中区域进行重绘，不进行字幕检测
        """
        from backend.inpaint.sttn_auto_inpaint import STTNAutoInpaint

        self.append_output(tr['Main']['ProcessingStartRemovingSubtitles'])
        mask_area_coordinates = []
        for sub_area in self.sub_areas:
            ymin, ymax, xmin, xmax = sub_area
            mask_area_coordinates.append((xmin, xmax, ymin, ymax))
        mask = create_mask(self.mask_size, mask_area_coordinates)
        with self.profile.measure("model_load"):
            sttn_video_inpaint = STTNAutoInpaint(self.hardware_accelerator.device, self.model_config.STTN_AUTO_MODEL_PATH, self.video_path)
        with self.profile.measure("model_inpaint"):
            sttn_video_inpaint(input_mask=mask, input_sub_remover=self, tbar=tbar)

    def video_inpaint(self, tbar, model):
        sub_detector = SubtitleDetect(self.video_path, self.sub_areas)
        with self.profile.measure("ocr_detect"):
            sub_list = sub_detector.find_subtitle_frame_no(sub_remover=self)
        self.profile.set("subtitle_detected_frames", len(sub_list))
        if len(sub_list) == 0:
            raise Exception(tr['Main']['NoSubtitleDetected'].format(self.video_path))
        with self.profile.measure("interval_processing"):
            continuous_frame_no_list = sub_detector.find_continuous_ranges_with_same_mask(sub_list)
            tbar.write(f"Subtitle detected: {continuous_frame_no_list}")
            continuous_frame_no_list = expand_frame_ranges(continuous_frame_no_list, config.subtitleTimelineBackwardFrameCount.value, config.subtitleTimelineForwardFrameCount.value)
            tbar.write(f"Subtitle timeline expand ({config.subtitleTimelineBackwardFrameCount.value} <- -> {config.subtitleTimelineForwardFrameCount.value}): {continuous_frame_no_list}")
            continuous_frame_no_list = sub_detector.filter_and_merge_intervals(continuous_frame_no_list, config.sttnReferenceLength.value)
            tbar.write(f'Subtitle filter_and_merge_intervals: {continuous_frame_no_list}')
        self.profile.set("subtitle_intervals", continuous_frame_no_list)
        del sub_detector
        gc.collect()
        start_end_map = dict()
        for start, end in continuous_frame_no_list:
            # 确保区间不超出视频总帧数，否则会导致 FramePrefetcher 哨兵被内循环消费后外层死锁
            start_end_map[start] = min(end, self.frame_count)
        current_frame_index = 0
        self.append_output(tr['Main']['ProcessingStartRemovingSubtitles'])
        # 使用帧预读取，I/O 与推理重叠
        reader = FramePrefetcher(self.video_cap)
        while True:
            ret, frame = reader.read()
            # 如果读取到为，则结束
            if not ret:
                break
            current_frame_index += 1
            # 判断当前帧号是不是字幕区间开始, 如果不是，则直接写
            if current_frame_index not in start_end_map.keys():
                self.video_writer.write(frame)
                # self.append_output(f'write frame: {current_frame_index}')
                self.update_progress(tbar, increment=1)
                self.update_preview_with_comp(frame, frame)
            # 如果是区间开始，则找到尾巴
            else:
                start_frame_index = current_frame_index
                end_frame_index = start_end_map[current_frame_index]
                tbar.write(f'processing frame {start_frame_index} to {end_frame_index}')
                # 用于存储需要去字幕的视频帧
                frames_need_inpaint = list()
                frames_need_inpaint.append(frame)
                inner_index = 0
                # 接着往下读，直到读取到尾巴
                for j in range(end_frame_index - start_frame_index):
                    ret, frame = reader.read()
                    if not ret:
                        break
                    current_frame_index += 1
                    frames_need_inpaint.append(frame)
                mask_area_coordinates = []
                # 1. 获取当前批次的mask坐标全集
                for mask_index in range(start_frame_index, end_frame_index):
                    if mask_index in sub_list.keys():
                        for area in sub_list[mask_index]:
                            xmin, xmax, ymin, ymax = area
                            # 判断是不是非字幕区域(如果宽大于长，则认为是错误检测)
                            if (ymax - ymin) - (xmax - xmin) > config.subtitleYXAxisDifferencePixel.value:
                                continue
                            mask_area = self.mask_area_for_mode(area)
                            if mask_area not in mask_area_coordinates:
                                mask_area_coordinates.append(mask_area)
                # 1. 获取当前批次使用的mask
                mask = create_mask(self.mask_size, mask_area_coordinates)
                # self.append_output(f'inpaint with mask: {mask_area_coordinates}')
                for batch in batch_generator(frames_need_inpaint, config.getSttnMaxLoadNum()):
                    # 2. 调用批推理
                    if len(batch) >= 1:
                        with self.profile.measure("model_inpaint"):
                            inpainted_frames = model(batch, mask)
                        for i, inpainted_frame in enumerate(inpainted_frames):
                            self.video_writer.write(inpainted_frame)
                            # self.append_output(f'write frame: {start_frame_index + inner_index} with mask')
                            inner_index += 1
                            self.update_preview_with_comp(np.clip(batch[i]+mask[:,:,np.newaxis]*0.3,0,255).astype(np.uint8), inpainted_frame)
                    self.update_progress(tbar, increment=len(batch))
        reader.stop()

    def finalize_profile(self, status, error=None):
        if self._profile_finalized:
            return
        self._profile_finalized = True
        self.profile.status = status
        if self._run_start_time is not None:
            self.profile.add_time("total", time.time() - self._run_start_time)
        else:
            self.profile.add_time("total", time.time() - self.profile.started_at)
        self.profile.set("output", os.path.abspath(self.video_out_path))
        self.profile.set("audio_merge_status", self.audio_merge_status)
        self.profile.set("model", config.inpaintMode.value.name.lower().replace("_", "-"))
        self.profile.set("mask_deviation", int(config.subtitleAreaDeviationPixel.value))
        self.profile.set("mask_mode", self.mask_mode.value if hasattr(self.mask_mode, "value") else self.mask_mode)
        self.profile.set("hardware_acceleration_enabled", bool(config.hardwareAcceleration.value))
        self.profile.set("accelerator", self.hardware_accelerator.accelerator_name)
        self.profile.set("mps_cpu_fallback_enabled", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1")
        self.profile.set("onnx_providers", list(self.hardware_accelerator.onnx_providers))
        self.profile.set("ocr_hpi_enabled", self.ocr_hpi_enabled())
        self.profile.set("ocr_detection_device", self.ocr_detection_device_name())
        if hasattr(self.video_writer, "frames_written"):
            self.profile.set("encoded_frames", self.video_writer.frames_written)
            self.profile.add_time("encode_write", self.video_writer.write_seconds)
            self.profile.add_time("encode_release", self.video_writer.release_seconds)
        if error is not None:
            self.profile.set("error", error)
        if self.profile_json_path:
            self.profile.write(self.profile_json_path)

    def run(self):
        self._run_start_time = time.time()
        tbar = None
        cap_released = False
        writer_released = False
        try:
            if len(self.sub_areas) == 0:
                self.sub_areas = self.default_subtitle_areas()
                if self.sub_areas:
                    self.append_output(f"Using default subtitle area: {self.sub_areas}")
                else:
                    self.append_output(tr['Main']['FullScreenProcessingNote'])
                    self.sub_areas.append((0, self.frame_height, 0, self.frame_width))
            self.append_output(tr['Main']['SubtitleArea'].format(self.sub_areas))
            self.append_output(tr['Main']['ABSection'].format(str(self.ab_sections).replace("range", "") if self.ab_sections is not None and len(self.ab_sections) > 0 else tr['Main']['ABSectionAll']))
            if self.hardware_accelerator.has_accelerator():
                accelerator_name = self.hardware_accelerator.accelerator_name
                if accelerator_name == 'DirectML' and config.inpaintMode.value not in [InpaintMode.STTN_AUTO, InpaintMode.STTN_DET]:
                    self.append_output(tr['Main']['DirectMLWarning'])
            self.video_out_path = os.path.abspath(self.video_out_path)
            os.makedirs(os.path.dirname(self.video_out_path), exist_ok=True)
            self.profile.set("subtitle_areas", self.sub_areas)
            self.progress_total = 0
            tbar = tqdm(total=int(self.frame_count), unit='frame', position=0, file=sys.__stdout__,
                        desc='Subtitle Removing')
            if self.is_picture:
                original_frame = read_image(self.video_path)
                if original_frame is None:
                    raise RuntimeError(tr['Main']['ReadImageFailed'].format(self.video_path))
                sub_detector = SubtitleDetect(self.video_path, self.sub_areas)
                with self.profile.measure("ocr_detect"):
                    sub_list = sub_detector.detect_subtitle(original_frame)
                self.profile.set("subtitle_detected_frames", 1 if len(sub_list) else 0)
                del sub_detector
                gc.collect()
                if len(sub_list):
                    mask = create_mask(original_frame.shape[0:2], sub_list)
                    with self.profile.measure("model_inpaint"):
                        inpainted_frame = self.lama_inpaint.inpaint(original_frame, mask)
                    self.update_preview_with_comp(np.clip(original_frame+mask[:,:,np.newaxis]*0.3,0,255).astype(np.uint8), inpainted_frame)
                else:
                    inpainted_frame = original_frame
                    self.update_preview_with_comp(original_frame, inpainted_frame)
                cv2.imencode(self.ext, inpainted_frame)[1].tofile(self.video_out_path)
                tbar.update(1)
                self.progress_total = 100
            else:
                self.validate_model_selection()
                self.log_model()
                if config.inpaintMode.value == InpaintMode.PROPAINTER:
                    self.propainter_mode(tbar)
                elif config.inpaintMode.value == InpaintMode.STTN_AUTO:
                    self.sttn_auto_mode(tbar)
                elif config.inpaintMode.value == InpaintMode.STTN_DET:
                    self.video_inpaint(tbar, self.sttn_det_inpaint)
                elif config.inpaintMode.value == InpaintMode.LAMA:
                    self.video_inpaint(tbar, self.lama_inpaint)
                elif config.inpaintMode.value == InpaintMode.OPENCV:
                    self.video_inpaint(tbar, OpenCVInpaint())
                else:
                    raise Exception(f'inpaint mode: {config.inpaintMode.value} not implemented')

            self.video_cap.release()
            cap_released = True
            self.video_writer.release()
            writer_released = True
            if not self.is_picture:
                self.merge_audio_to_video()
            self.append_output(tr['Main']['FinishedProcessing'].format(self.video_out_path))
            self.append_output(tr['Main']['ProcessingTime'].format(round(time.time() - self._run_start_time)))
            self.isFinished = True
            self.progress_total = 100
            self.finalize_profile("success")
        except Exception as exc:
            self.finalize_profile("failed", error="".join(traceback.format_exception_only(type(exc), exc)).strip())
            raise
        finally:
            if tbar is not None:
                tbar.close()
            if not cap_released:
                self.video_cap.release()
            if not writer_released:
                try:
                    self.video_writer.release()
                except Exception as cleanup_exc:
                    self.profile.add_event("video_writer_cleanup_failed", error=str(cleanup_exc))
            if os.path.exists(self.video_temp_file.name):
                try:
                    os.remove(self.video_temp_file.name)
                except Exception:
                    pass

    def log_model(self):
        model_friendly_name = list(tr['InpaintMode'].values())[list(InpaintMode).index(config.inpaintMode.value)]
        model_device = 'CPU'
        if config.inpaintMode.value != InpaintMode.OPENCV and self.hardware_accelerator.has_accelerator():
            accelerator_name = self.hardware_accelerator.accelerator_name
            if accelerator_name == 'DirectML' and config.inpaintMode.value in [InpaintMode.STTN_AUTO, InpaintMode.STTN_DET]:
                model_device = 'DirectML'
            if self.hardware_accelerator.has_cuda() or self.hardware_accelerator.has_mps():
                model_device = accelerator_name
        self.append_output(tr['Main']['SubtitleRemoverModel'].format(f"{model_friendly_name} ({model_device})"))
        detect_mode_name = list(tr['SubtitleDetectMode'].values())[list(SubtitleDetectMode).index(config.subtitleDetectMode.value)]
        self.append_output(tr['Main']['SubtitleDetectionModel'].format(
            f"{detect_mode_name} ({self.ocr_detection_device_name()})"
        ))

    def validate_model_selection(self):
        mode = config.inpaintMode.value
        if mode != InpaintMode.OPENCV:
            self.profile.add_event(
                "slow_model_selected",
                model=mode.value,
                allow_slow_models=self.allow_slow_models,
            )
        if self.allow_slow_models:
            return
        if sys.platform == "darwin" and mode == InpaintMode.PROPAINTER:
            raise RuntimeError(
                "ProPainter is disabled by default on macOS because this machine measured "
                "about 159s for 48 synthetic frames and timed out at 900s on the real test "
                "when PyTorch fell back from unsupported MPS ops to CPU. Use OpenCV for the "
                "Mac app, or rerun CLI with --allow-slow-models for an explicit experiment."
            )
        if mode == InpaintMode.LAMA and not self.hardware_accelerator.has_accelerator():
            raise RuntimeError(
                "LAMA CPU is disabled by default because it timed out at 600s on the real "
                "test. Enable GPU/MPS acceleration or rerun CLI with --allow-slow-models "
                "for an explicit experiment."
            )

    def ocr_hpi_enabled(self):
        onnx_providers = self.hardware_accelerator.onnx_providers
        hpi_providers = [provider for provider in onnx_providers if provider != "CPUExecutionProvider"]
        return bool(
            self.hardware_accelerator.has_accelerator()
            and len(hpi_providers) > 0
            and paddle_hpi_available()
        )

    def ocr_detection_device_name(self):
        if self.ocr_hpi_enabled():
            return ", ".join(self.hardware_accelerator.onnx_providers)
        return "CPU"

    def mask_area_for_mode(self, area):
        mask_mode = self.mask_mode.value if hasattr(self.mask_mode, "value") else self.mask_mode
        if mask_mode == "box":
            return area
        xmin, xmax, ymin, ymax = area
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        for sub_ymin, sub_ymax, sub_xmin, sub_xmax in self.sub_areas:
            if sub_ymin <= center_y <= sub_ymax and sub_xmin <= center_x <= sub_xmax:
                if mask_mode == "line":
                    return (sub_xmin, sub_xmax, ymin, ymax)
                if mask_mode == "area":
                    return (sub_xmin, sub_xmax, sub_ymin, sub_ymax)
        if mask_mode == "line":
            return (0, self.frame_width, ymin, ymax)
        if mask_mode == "area" and self.sub_areas:
            sub_ymin, sub_ymax, sub_xmin, sub_xmax = self.sub_areas[0]
            return (sub_xmin, sub_xmax, sub_ymin, sub_ymax)
        return area

    def merge_audio_to_video(self):
        ffmpeg_path = FFmpegCLI.instance().ffmpeg_path
        use_shell = True if os.name == "nt" else False
        probe_command = [ffmpeg_path, "-hide_banner", "-i", self.video_path]
        with self.profile.measure("audio_probe"):
            probe = subprocess.run(probe_command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, shell=use_shell, timeout=60)
        probe_text = (probe.stdout or b"") + (probe.stderr or b"")
        has_audio = b"Audio:" in probe_text
        if not has_audio:
            self.audio_merge_status = "no_audio"
            with self.profile.measure("audio_fallback_copy"):
                shutil.copy2(self.video_temp_file.name, self.video_out_path)
            self.video_temp_file.close()
            return

        temp = tempfile.NamedTemporaryFile(suffix='.m4a', delete=False)
        temp_path = temp.name
        temp.close()
        audio_extract_command = [ffmpeg_path,
                                 "-y", "-i", self.video_path,
                                 "-map", "0:a:0",
                                 "-acodec", "copy",
                                 "-vn", "-loglevel", "error", temp_path]
        try:
            with self.profile.measure("audio_extract"):
                extract = subprocess.run(audio_extract_command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, shell=use_shell, timeout=600)
            if extract.returncode != 0:
                stderr = (extract.stderr or b"").decode("utf-8", errors="replace").strip()
                self.audio_merge_status = "extract_failed_fallback_video_only"
                with self.profile.measure("audio_fallback_copy"):
                    shutil.copy2(self.video_temp_file.name, self.video_out_path)
                raise RuntimeError(f"Audio extract failed; wrote video-only fallback to {self.video_out_path}: {stderr}")
            if os.path.exists(self.video_temp_file.name):
                audio_merge_command = [ffmpeg_path,
                                       "-y", "-i", self.video_temp_file.name,
                                       "-i", temp_path,
                                       "-vcodec", "copy",
                                       "-acodec", "copy",
                                       "-loglevel", "error", self.video_out_path]
                with self.profile.measure("audio_merge"):
                    merge = subprocess.run(audio_merge_command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, shell=use_shell, timeout=600)
                if merge.returncode != 0:
                    stderr = (merge.stderr or b"").decode("utf-8", errors="replace").strip()
                    self.is_successful_merged = False
                    self.audio_merge_status = "merge_failed_fallback_video_only"
                    with self.profile.measure("audio_fallback_copy"):
                        shutil.copy2(self.video_temp_file.name, self.video_out_path)
                    raise RuntimeError(f"Audio merge failed; wrote video-only fallback to {self.video_out_path}: {stderr}")
                self.is_successful_merged = True
                self.audio_merge_status = "success"
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            self.video_temp_file.close()

    @cached_property
    def lama_inpaint(self):
        from backend.inpaint.lama_inpaint import LamaInpaint

        model_path = self.model_config.ensure_lama_model()
        device = self.hardware_accelerator.device if self.hardware_accelerator.has_cuda() or self.hardware_accelerator.has_mps() else cpu_torch_device()
        with self.profile.measure("model_load"):
            return LamaInpaint(device, model_path)

    @cached_property
    def sttn_det_inpaint(self):
        from backend.inpaint.sttn_det_inpaint import STTNDetInpaint

        with self.profile.measure("model_load"):
            return STTNDetInpaint(self.hardware_accelerator.device, self.model_config.STTN_DET_MODEL_PATH)

    def default_subtitle_areas(self):
        areas = []
        areas_str = config.subtitleSelectionAreas.value
        if not areas_str:
            return areas
        for area in areas_str.split(";"):
            try:
                ymin_r, ymax_r, xmin_r, xmax_r = map(float, area.split(","))
            except ValueError:
                continue
            ymin = round(max(0.0, min(ymin_r, 1.0)) * self.frame_height)
            ymax = round(max(0.0, min(ymax_r, 1.0)) * self.frame_height)
            xmin = round(max(0.0, min(xmin_r, 1.0)) * self.frame_width)
            xmax = round(max(0.0, min(xmax_r, 1.0)) * self.frame_width)
            if ymax > ymin and xmax > xmin:
                areas.append((ymin, ymax, xmin, xmax))
        return areas

def main(argv=None):
    multiprocessing.set_start_method("spawn", force=True)
    from backend.tools.args_handler import parse_args
    args = parse_args(argv)
    # force english
    config.set(config.interface, 'en')
    TRANSLATION_FILE = os.path.join(BASE_DIR, 'interface', f"{config.interface.value}.ini")
    tr.read(TRANSLATION_FILE, encoding='utf-8')
    if args.no_gpu:
        config.hardwareAcceleration.value = False
    if args.detect_fps is not None:
        config.subtitleDetectionSampleFps.value = args.detect_fps
    if args.ocr_max_dim is not None:
        config.subtitleDetectionMaxDimension.value = args.ocr_max_dim
    if args.mask_deviation is not None:
        config.subtitleAreaDeviationPixel.value = args.mask_deviation
    sr = SubtitleRemover(args.input)
    sr.profile_json_path = args.profile_json
    sr.allow_slow_models = args.allow_slow_models
    sr.mask_mode = args.mask_mode
    if not is_video_or_image(args.input):
        error = f'Error: {args.input} is not supported or corrupted.'
        sr.append_output(error)
        sr.finalize_profile("failed", error=error)
        return 1
    if args.subtitle_area_ratio:
        sr.sub_areas = [
            (
                round(max(0.0, min(ymin, 1.0)) * sr.frame_height),
                round(max(0.0, min(ymax, 1.0)) * sr.frame_height),
                round(max(0.0, min(xmin, 1.0)) * sr.frame_width),
                round(max(0.0, min(xmax, 1.0)) * sr.frame_width),
            )
            for ymin, ymax, xmin, xmax in args.subtitle_area_ratio
        ]
    else:
        sr.sub_areas = args.subtitle_area_coords
    if args.output:
        sr.video_out_path = os.path.abspath(args.output)
    config.inpaintMode.value = args.inpaint_mode
    sr.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
        
