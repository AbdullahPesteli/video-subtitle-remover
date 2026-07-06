import sys
from functools import cached_property, lru_cache

import cv2
from tqdm import tqdm

from .model_config import ModelConfig
from .hardware_accelerator import HardwareAccelerator
from .common_tools import get_readable_path
from .ocr import get_coordinates
from backend.config import config, tr
from backend.scenedetect import scene_detect
from backend.scenedetect.detectors import ContentDetector
from backend.tools.inpaint_tools import is_frame_number_in_ab_sections


@lru_cache(maxsize=1)
def paddle_hpi_available():
    try:
        from paddlex.utils.deps import require_hpip
        require_hpip()
        return True
    except Exception:
        return False


class SubtitleDetect:
    """
    文本框检测类，用于检测视频帧中是否存在文本框
    """

    # 采样间隔，根据视频帧率在 _init_sample_step 中自适应设置
    SAMPLE_STEP = 3

    def __init__(self, video_path, sub_areas=[]):
        self.video_path = video_path
        self.sub_areas = sub_areas
        self._init_sample_step()

    def _init_sample_step(self):
        """根据视频帧率自适应设置采样间隔。"""
        cap = cv2.VideoCapture(get_readable_path(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        target_fps = max(1, int(config.subtitleDetectionSampleFps.value))
        if fps and fps > 0:
            self.SAMPLE_STEP = max(1, int(round(fps / target_fps)))
        else:
            self.SAMPLE_STEP = 3

    @cached_property
    def text_detector(self):
        import paddle
        paddle.disable_signal_handler()
        from paddleocr import TextDetection
        hardware_accelerator = HardwareAccelerator.instance()
        onnx_providers = hardware_accelerator.onnx_providers
        hpi_providers = [provider for provider in onnx_providers if provider != "CPUExecutionProvider"]
        enable_hpi = (
            hardware_accelerator.has_accelerator()
            and len(hpi_providers) > 0
            and paddle_hpi_available()
        )
        model_config = ModelConfig()
        return TextDetection(
            model_name=model_config.DET_MODEL_NAME,
            model_dir=model_config.DET_MODEL_DIR,
            device="cpu",
            enable_hpi=enable_hpi,
        )

    @staticmethod
    def _clip_area(area, width, height):
        ymin, ymax, xmin, xmax = area
        ymin = max(0, min(int(round(ymin)), height))
        ymax = max(0, min(int(round(ymax)), height))
        xmin = max(0, min(int(round(xmin)), width))
        xmax = max(0, min(int(round(xmax)), width))
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymax <= ymin or xmax <= xmin:
            return None
        return ymin, ymax, xmin, xmax

    @staticmethod
    def _dedupe_regions(regions):
        seen = set()
        deduped = []
        for region in regions:
            if region in seen:
                continue
            seen.add(region)
            deduped.append(region)
        return deduped

    def _prepare_ocr_image(self, img):
        max_dim = max(480, int(config.subtitleDetectionMaxDimension.value))
        height, width = img.shape[:2]
        largest = max(height, width)
        if largest <= max_dim:
            return img, 1.0
        scale = max_dim / float(largest)
        resized = cv2.resize(
            img,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    def _detect_text_regions(self, img, x_offset=0, y_offset=0):
        temp_list = []
        ocr_img, scale = self._prepare_ocr_image(img)
        results = self.text_detector.predict(ocr_img)
        inv_scale = 1.0 / scale
        for res in results:
            dt_polys = res['dt_polys']
            if dt_polys is None or len(dt_polys) == 0:
                continue
            coordinate_list = get_coordinates(dt_polys.tolist())
            for xmin, xmax, ymin, ymax in coordinate_list:
                if scale != 1.0:
                    xmin = int(round(xmin * inv_scale))
                    xmax = int(round(xmax * inv_scale))
                    ymin = int(round(ymin * inv_scale))
                    ymax = int(round(ymax * inv_scale))
                temp_list.append((xmin + x_offset, xmax + x_offset, ymin + y_offset, ymax + y_offset))
        return temp_list

    def detect_subtitle(self, img):
        temp_list = []
        sub_areas = self.sub_areas
        has_areas = sub_areas is not None and len(sub_areas) > 0
        if not has_areas:
            return self._detect_text_regions(img)

        height, width = img.shape[:2]
        for area in sub_areas:
            clipped = self._clip_area(area, width, height)
            if clipped is None:
                continue
            ymin, ymax, xmin, xmax = clipped
            crop = img[ymin:ymax, xmin:xmax]
            temp_list.extend(self._detect_text_regions(crop, x_offset=xmin, y_offset=ymin))
        return self._dedupe_regions(temp_list)

    def find_subtitle_frame_no(self, sub_remover=None):
        video_cap = cv2.VideoCapture(get_readable_path(self.video_path))
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        tbar = tqdm(total=int(frame_count), unit='frame', position=0, file=sys.__stdout__, desc='Subtitle Finding')
        current_frame_no = 0
        # 阶段1：采样检测，仅对每隔 sample_step 帧执行 OCR
        sampled_results = {}  # frame_no -> temp_list
        if sub_remover:
            sub_remover.append_output(tr['Main']['ProcessingStartFindingSubtitles'])
        ab_sections = sub_remover.ab_sections if sub_remover is not None else None
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            # 如果读取视频帧失败（视频读到最后一帧）
            if not ret:
                break
            # 读取视频帧成功
            current_frame_no += 1
            if not is_frame_number_in_ab_sections(current_frame_no - 1, ab_sections):
                tbar.update(1)
                continue
            # 仅对采样帧执行 OCR 推理
            if (current_frame_no - 1) % self.SAMPLE_STEP == 0 or self.SAMPLE_STEP <= 1:
                temp_list = self.detect_subtitle(frame)
                if len(temp_list) > 0:
                    sampled_results[current_frame_no] = temp_list
            tbar.update(1)
            if sub_remover:
                sub_remover.progress_total = (100 * float(current_frame_no) / float(frame_count)) // 2
        video_cap.release()
        # 阶段2：插值填充 — 两个采样帧之间都有字幕时，中间帧也标记为有字幕
        subtitle_frame_no_box_dict = {}
        detected_nos = sorted(sampled_results.keys())
        max_gap = self.SAMPLE_STEP * 2
        for f, next_f in zip(detected_nos, detected_nos[1:]):
            subtitle_frame_no_box_dict[f] = sampled_results[f]
            if next_f - f <= max_gap:
                fill_mask = sampled_results[f]
                for fill_f in range(f + 1, next_f):
                    subtitle_frame_no_box_dict[fill_f] = fill_mask
        # 添加最后一个检测帧
        if detected_nos:
            subtitle_frame_no_box_dict[detected_nos[-1]] = sampled_results[detected_nos[-1]]
        subtitle_frame_no_box_dict = self.unify_regions(subtitle_frame_no_box_dict)
        if sub_remover:
            sub_remover.append_output(tr['Main']['FinishedFindingSubtitles'])
        new_subtitle_frame_no_box_dict = dict()
        for key in subtitle_frame_no_box_dict.keys():
            if len(subtitle_frame_no_box_dict[key]) > 0:
                new_subtitle_frame_no_box_dict[key] = subtitle_frame_no_box_dict[key]
        return new_subtitle_frame_no_box_dict

    @staticmethod
    def split_range_by_scene(intervals, points):
        # 确保离散值列表是有序的
        points.sort()
        # 用于存储结果区间的列表
        result_intervals = []
        # 遍历区间
        for start, end in intervals:
            # 在当前区间内的点
            current_points = [p for p in points if start <= p <= end]

            # 遍历当前区间内的离散点
            for p in current_points:
                # 如果当前离散点不是区间的起始点，添加从区间开始到离散点前一个数字的区间
                if start < p:
                    result_intervals.append((start, p - 1))
                # 更新区间开始为当前离散点
                start = p
            # 添加从最后一个离散点或区间开始到区间结束的区间
            result_intervals.append((start, end))
        # 输出结果
        return result_intervals

    @staticmethod
    def get_scene_div_frame_no(v_path):
        """
        获取发生场景切换的帧号
        """
        scene_div_frame_no_list = []
        scene_list = scene_detect(v_path, ContentDetector())
        for scene in scene_list:
            start, end = scene
            if start.frame_num == 0:
                pass
            else:
                scene_div_frame_no_list.append(start.frame_num + 1)
        return scene_div_frame_no_list

    @staticmethod
    def are_similar(region1, region2):
        """判断两个区域是否相似。"""
        xmin1, xmax1, ymin1, ymax1 = region1
        xmin2, xmax2, ymin2, ymax2 = region2

        return abs(xmin1 - xmin2) <= config.subtitleAreaPixelToleranceXPixel.value and abs(xmax1 - xmax2) <= config.subtitleAreaPixelToleranceXPixel.value and \
            abs(ymin1 - ymin2) <= config.subtitleAreaPixelToleranceYPixel.value and abs(ymax1 - ymax2) <= config.subtitleAreaPixelToleranceYPixel.value

    def unify_regions(self, raw_regions):
        """将连续相似的区域统一，保持列表结构。"""
        if len(raw_regions) > 0:
            keys = sorted(raw_regions.keys())  # 对键进行排序以确保它们是连续的
            unified_regions = {}

            # 初始化
            last_key = keys[0]
            unify_value_map = {last_key: raw_regions[last_key]}

            for key in keys[1:]:
                current_regions = raw_regions[key]

                # 新增一个列表来存放匹配过的标准区间
                new_unify_values = []

                for idx, region in enumerate(current_regions):
                    last_standard_region = unify_value_map[last_key][idx] if idx < len(unify_value_map[last_key]) else None

                    # 如果当前的区间与前一个键的对应区间相似，我们统一它们
                    if last_standard_region and self.are_similar(region, last_standard_region):
                        new_unify_values.append(last_standard_region)
                    else:
                        new_unify_values.append(region)

                # 更新unify_value_map为最新的区间值
                unify_value_map[key] = new_unify_values
                last_key = key

            # 将最终统一后的结果传递给unified_regions
            for key in keys:
                unified_regions[key] = unify_value_map[key]
            return unified_regions
        else:
            return raw_regions

    @staticmethod
    def find_continuous_ranges(subtitle_frame_no_box_dict):
        """
        获取字幕出现的起始帧号与结束帧号
        """
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 初始区间开始值

        for i in range(1, len(numbers)):
            # 如果当前数字与前一个数字间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict):
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 初始区间开始值
        for i in range(1, len(numbers)):
            # 如果当前帧号与前一个帧号间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
            # 如果当前帧号与前一个帧号间隔为1，且当前帧号对应的坐标点与上一帧号对应的坐标点不一致
            # 记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] == 1:
                if subtitle_frame_no_box_dict[numbers[i]] != subtitle_frame_no_box_dict[numbers[i - 1]]:
                    end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                    ranges.append((start, end))
                    start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def filter_and_merge_intervals(intervals, target_length):
        """
        合并传入的字幕起始区间，确保区间大小最低为STTN_REFERENCE_LENGTH
        复杂度 O(n log n)
        """
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        # 一次遍历：扩展单点区间，利用排序后的相邻关系 O(n)
        expanded = []
        for i, (start, end) in enumerate(intervals):
            if start == end:  # 单点区间
                prev_end = expanded[-1][1] if expanded else float('-inf')
                next_start = intervals[i + 1][0] if i + 1 < len(intervals) else float('inf')
                half = (target_length - 1) // 2
                new_start = max(start - half, prev_end + 1)
                new_end = min(start + half, next_start - 1)
                if new_end < new_start:
                    new_start, new_end = start, start
                expanded.append((new_start, new_end))
            else:
                expanded.append((start, end))
        # 一次遍历：合并重叠或相邻的短区间 O(n)
        merged = [expanded[0]]
        for start, end in expanded[1:]:
            last_start, last_end = merged[-1]
            last_len = last_end - last_start + 1
            cur_len = end - start + 1
            if (start <= last_end or start == last_end + 1) and (cur_len < target_length or last_len < target_length):
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged
