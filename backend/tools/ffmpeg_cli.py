import os
import stat
import sys

import platform
from .common_tools import merge_big_file_if_not_exists
from backend.config import BASE_DIR

class FFmpegCLI:
    
    """
    进程管理器类，用于管理子进程的生命周期
    使用弱引用避免内存泄漏
    """
    _instance = None
    
    @classmethod
    def instance(cls):
        """单例模式获取实例"""
        if cls._instance is None:
            cls._instance = FFmpegCLI()
        return cls._instance
    
    def __init__(self):
        os.chmod(self.ffmpeg_path, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
        
    @property
    def ffmpeg_path(self):
        system = platform.system()
        if system == "Windows":
            ffmpeg_dir = self._find_ffmpeg_dir('win_x64')
            merge_big_file_if_not_exists(ffmpeg_dir, 'ffmpeg.exe')
            return os.path.join(ffmpeg_dir, 'ffmpeg.exe')
        elif system == "Linux":
            return os.path.join(self._find_ffmpeg_dir('linux_x64'), 'ffmpeg')
        else:
            return os.path.join(self._find_ffmpeg_dir('macos'), 'ffmpeg')

    def _find_ffmpeg_dir(self, platform_dir):
        candidates = [os.path.join(BASE_DIR, 'ffmpeg', platform_dir)]
        if getattr(sys, 'frozen', False):
            executable_dir = os.path.dirname(sys.executable)
            candidates.extend([
                os.path.abspath(os.path.join(executable_dir, '..', 'Resources', 'backend', 'ffmpeg', platform_dir)),
                os.path.abspath(os.path.join(executable_dir, '..', 'Frameworks', 'backend', 'ffmpeg', platform_dir)),
            ])
            if hasattr(sys, '_MEIPASS'):
                candidates.append(os.path.join(sys._MEIPASS, 'backend', 'ffmpeg', platform_dir))
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]
