#!coding=utf-8

import json 
import os
import pickle
import gzip
from .util import read_video

class VideoTemplateCache:
    def __init__(self, video_info_config_path, video_template_dir, video_feature_dir, mask_base_dir):
        with open(video_info_config_path, 'r') as fp:
            self.video_infos = json.load(fp)

        self.video_template_dir = video_template_dir
        self.video_feature_dir = video_feature_dir
        self.mask_base_dir = mask_base_dir

        self._dummy_data = {
            'video_frames': None,
            'mask_video_frames': None,
            'video_feature': None
        }

        self._cache = {}
    
    def get_cache_data(self, video_name: str, debug=False):
        if debug:
            print(f"[DEBUG] get cache data from {video_name}")
        return self._cache.get(video_name, self._dummy_data)
    
    def _load_video_template(self, video_name: str, debug=False):
        video_template_filepath = os.path.join(self.video_template_dir, f"{video_name}.mp4")

        if not os.path.exists(video_template_filepath):
            return None

        if debug:
            print(f"🔄 加载缓存视频: {video_template_filepath}")

        video_frames = read_video(video_template_filepath, use_decord=False, change_fps=False)
        return video_frames, video_template_filepath
    
    def _load_mask_template(self, video_name: str, debug=False):
        if self.mask_base_dir is None:
            return None

        mask_template_filepath = os.path.join(self.mask_base_dir, f"{video_name}_mask.mp4")

        if not os.path.exists(mask_template_filepath):
            return None
        
        if debug:
            print(f"🎭 读取mask视频帧: {mask_template_filepath}")

        mask_video_frames = read_video(mask_template_filepath, use_decord=False, change_fps=False)
        return mask_video_frames, mask_template_filepath
    
    def _load_video_feature(self, video_name: str, debug=False):
        video_feature_filepath = os.path.join(self.video_feature_dir, f"{video_name}_features.pkl")
        if not os.path.exists(video_feature_filepath):
            return None
            
        if debug:
            print(f"🔄 加载缓存特征: {video_feature_filepath}")

        with gzip.open(video_feature_filepath, 'rb') as f:
            video_feature = pickle.load(f)
            return video_feature, video_feature_filepath
    

    def load_all_data(self, debug=False):
        for video_name in self.video_infos.keys():
            video_frames, video_feature_filepath = self._load_video_template(video_name, debug)
            mask_video_frames, mask_template_filepath = self._load_mask_template(video_name, debug)
            video_feature, video_template_filepath = self._load_video_feature(video_name, debug)

            self._cache[video_name] = {
                'video_frames': video_frames,
                'mask_video_frames': mask_video_frames,
                'video_feature': video_feature
            }
