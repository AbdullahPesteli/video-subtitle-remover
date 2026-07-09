import os
from backend.config import config, BASE_DIR
from backend.tools.common_tools import merge_big_file_if_not_exists
from backend.tools.constant import SubtitleDetectMode

_MODEL_NAME_MAP = {
    SubtitleDetectMode.PP_OCRv5_MOBILE: "PP-OCRv5_mobile_det",
    SubtitleDetectMode.PP_OCRv5_SERVER: "PP-OCRv5_server_det",
}

class ModelConfig:
    def __init__(self):
        self.LAMA_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'big-lama')
        self.LAMA_MODEL_PATH = os.path.join(self.LAMA_MODEL_DIR, 'big-lama.pt')
        self.STTN_AUTO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn-auto', 'infer_model.pth')
        self.STTN_DET_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn-det', 'sttn.pth')
        self.PROPAINTER_MODEL_DIR = os.path.join(BASE_DIR,'models', 'propainter')
        self.PROPAINTER_MODEL_PATH = os.path.join(self.PROPAINTER_MODEL_DIR, 'ProPainter.pth')
        if config.subtitleDetectMode.value == SubtitleDetectMode.PP_OCRv5_MOBILE:
            self.DET_MODEL_DIR = os.path.join(BASE_DIR,'models', 'V5', 'ch_det_fast')
        elif config.subtitleDetectMode.value == SubtitleDetectMode.PP_OCRv5_SERVER:
            self.DET_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'V5', 'ch_det')
        else:
            raise ValueError(f"Invalid subtitle detect mode: {config.subtitleDetectMode.value}")
        self.DET_MODEL_NAME = _MODEL_NAME_MAP[config.subtitleDetectMode.value]

    @staticmethod
    def _ensure_merged_model(model_dir, model_file):
        merge_big_file_if_not_exists(model_dir, model_file)
        model_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Required model file was not created after merging split files: {model_path}"
            )
        return model_path

    def ensure_lama_model(self):
        return self._ensure_merged_model(self.LAMA_MODEL_DIR, 'big-lama.pt')

    def ensure_propainter_model(self):
        return self._ensure_merged_model(self.PROPAINTER_MODEL_DIR, 'ProPainter.pth')
