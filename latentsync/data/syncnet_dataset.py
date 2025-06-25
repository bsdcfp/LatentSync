# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from torch.utils.data import Dataset
import torch
import random
from ..utils.util import gather_video_paths_recursively
from ..utils.image_processor import ImageProcessor
from ..utils.audio import melspectrogram
import math
from einops import rearrange

from decord import AudioReader, VideoReader, cpu
import cv2


class SyncNetDataset(Dataset):
    def __init__(self, data_dir: str, fileslist: str, config):
        if fileslist != "":
            with open(fileslist) as file:
                self.video_paths = [line.rstrip() for line in file]
        elif data_dir != "":
            self.video_paths = gather_video_paths_recursively(data_dir)
        else:
            raise ValueError("data_dir and fileslist cannot be both empty")

        self.resolution = config.data.resolution
        self.num_frames = config.data.num_frames

        self.mel_window_length = math.ceil(self.num_frames / 5 * 16)

        self.audio_sample_rate = config.data.audio_sample_rate
        self.video_fps = config.data.video_fps
        self.audio_samples_length = int(
            config.data.audio_sample_rate // config.data.video_fps * config.data.num_frames
        )
        self.image_processor = ImageProcessor(resolution=config.data.resolution, mask="half")
        self.audio_mel_cache_dir = config.data.audio_mel_cache_dir
        os.makedirs(self.audio_mel_cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.video_paths)

    def read_audio(self, video_path: str):
        ar = AudioReader(video_path, ctx=cpu(self.worker_id), sample_rate=self.audio_sample_rate)
        original_mel = melspectrogram(ar[:].asnumpy().squeeze(0))
        return torch.from_numpy(original_mel)

    def crop_audio_window(self, original_mel, start_index):
        start_idx = int(80.0 * (start_index / float(self.video_fps)))
        end_idx = start_idx + self.mel_window_length
        return original_mel[:, start_idx:end_idx].unsqueeze(0)

    def get_frames(self, video_reader: VideoReader):
        total_num_frames = len(video_reader)

        start_idx = random.randint(0, total_num_frames - self.num_frames)
        frames_index = np.arange(start_idx, start_idx + self.num_frames, dtype=int)

        while True:
            wrong_start_idx = random.randint(0, total_num_frames - self.num_frames)
            # wrong_start_idx = random.randint(
            #     max(0, start_idx - 25), min(total_num_frames - self.num_frames, start_idx + 25)
            # )
            if wrong_start_idx == start_idx:
                continue
            # if wrong_start_idx >= start_idx - self.num_frames and wrong_start_idx <= start_idx + self.num_frames:
            #     continue
            wrong_frames_index = np.arange(wrong_start_idx, wrong_start_idx + self.num_frames, dtype=int)
            break

        frames = video_reader.get_batch(frames_index).asnumpy()
        wrong_frames = video_reader.get_batch(wrong_frames_index).asnumpy()

        return frames, wrong_frames, start_idx

    def worker_init_fn(self, worker_id):
        # Initialize the face mesh object in each worker process,
        # because the face mesh object cannot be called in subprocesses
        self.worker_id = worker_id
        # setattr(self, f"image_processor_{worker_id}", ImageProcessor(self.resolution, self.mask))

    def __getitem__(self, idx):
        # image_processor = getattr(self, f"image_processor_{self.worker_id}")
        while True:
            try:
                idx = random.randint(0, len(self) - 1)

                # Get video file path
                video_path = self.video_paths[idx]

                vr = VideoReader(video_path, ctx=cpu(self.worker_id))

                if len(vr) < 2 * self.num_frames:
                    continue

                frames, wrong_frames, start_idx = self.get_frames(vr)

                mel_cache_path = os.path.join(
                    self.audio_mel_cache_dir, os.path.basename(video_path).replace(".mp4", "_mel.pt")
                )

                if os.path.isfile(mel_cache_path):
                    try:
                        original_mel = torch.load(mel_cache_path)
                    except Exception as e:
                        print(f"{type(e).__name__} - {e} - {mel_cache_path}")
                        os.remove(mel_cache_path)
                        original_mel = self.read_audio(video_path)
                        torch.save(original_mel, mel_cache_path)
                else:
                    original_mel = self.read_audio(video_path)
                    torch.save(original_mel, mel_cache_path)

                mel = self.crop_audio_window(original_mel, start_idx)

                if mel.shape[-1] != self.mel_window_length:
                    continue

                if random.choice([True, False]):
                    y = torch.ones(1).float()
                    chosen_frames = frames
                else:
                    y = torch.zeros(1).float()
                    chosen_frames = wrong_frames

                chosen_frames = self.image_processor.process_images(chosen_frames)
                # chosen_frames, _, _ = image_processor.prepare_masks_and_masked_images(
                #     chosen_frames, affine_transform=True
                # )

                vr.seek(0)  # avoid memory leak
                break

            except Exception as e:  # Handle the exception of face not detcted
                print(f"{type(e).__name__} - {e} - {video_path}")
                if "vr" in locals():
                    vr.seek(0)  # avoid memory leak

        sample = dict(frames=chosen_frames, audio_samples=mel, y=y)

        return sample




class SyncNetDataset_v1(Dataset):
    """
    SyncNet数据集 - 基于SyncNet模型需求设计，同时适配自定义数据处理逻辑
    与原始SyncNetDataset保持一致的参数设置方式
    """
    def __init__(self, train_data_dir: list, config):
        """
        初始化SyncNet数据集
        
        Args:
            train_data_dir: 训练数据根目录列表
            config: 配置对象
        """
        if isinstance(train_data_dir, str):
            train_data_dir = [train_data_dir]  # 兼容单目录情况
        
        self.train_data_dirs = train_data_dir
        self.sequences = []
        self.sequence_to_data_dir = {}  # 记录每个序列对应的数据目录
        
        # 使用线程池并行扫描所有数据目录
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(self.train_data_dirs), 8)) as executor:
            results = list(executor.map(self._scan_data_dir, self.train_data_dirs))
        
        # 合并结果
        for seq_list, seq_to_dir in results:
            self.sequences.extend(seq_list)
            self.sequence_to_data_dir.update(seq_to_dir)
        
        if len(self.sequences) == 0:
            raise ValueError("未找到有效的序列列表")
        
        # 基本参数设置 - 与原始SyncNetDataset保持一致
        self.resolution = config.data.resolution
        self.num_frames = config.data.num_frames
        
        # mel频谱图窗口长度计算方式与原始一致
        self.mel_window_length = math.ceil(self.num_frames / 5 * 16)
        
        self.audio_sample_rate = config.data.audio_sample_rate
        self.video_fps = config.data.video_fps
        self.audio_samples_length = int(
            config.data.audio_sample_rate // config.data.video_fps * config.data.num_frames
        )
        
        # 图像处理器设置与原始一致
        self.image_processor = ImageProcessor(resolution=config.data.resolution, mask="half")
        
        # 音频处理相关
        self.audio_mel_cache_dir = config.data.audio_mel_cache_dir
        os.makedirs(self.audio_mel_cache_dir, exist_ok=True)
        
        # # 特有设置
        # self.mask_path = "/home/work/houyi/pj_25_q1/LatentSync/hy_test/crop_face/narrowed_mask/output_mask_09.png"
        # self.mask_image = load_fixed_mask(self.resolution, mask_path=self.mask_path)
        
        # 裁剪比例，与crop_face_no_margin_video_v2.py保持一致
        self.crop_scale = 0.78125  # 默认值 200/256
        # self.crop_scale = 1 
        self.width_scale = 1.
        # self.width_scale = 1.
        
        # 黑名单和缓存
        self.frame_paths_cache = {}
        self.blacklist_sequences = set()  # 存储问题序列
        
        print(f"初始化SyncNetDataset_v1, 总共找到 {len(self.sequences)} 个序列")
        print(f"帧数: {self.num_frames}, 分辨率: {self.resolution}, mel长度: {self.mel_window_length}")
        print(f"使用裁剪比例: {self.crop_scale}")
        # print(f"使用mask路径: {self.mask_path}")
    
    def _scan_data_dir(self, data_dir):
        """
        扫描单个数据目录并返回序列列表
        """
        seq_list = []
        seq_to_dir = {}
        
        frames_dir = os.path.join(data_dir, "align_frame")
        audios_dir = os.path.join(data_dir, "audios")
        
        dir_sequence_count = 0  # 记录当前数据目录中找到的序列数量
        
        if os.path.exists(frames_dir) and os.path.exists(audios_dir):
            for seq_dir in os.listdir(frames_dir):
                seq_path = os.path.join(frames_dir, seq_dir)
                audio_path = os.path.join(audios_dir, f"{seq_dir}.wav")
                
                # 检查是否同时存在帧目录和对应的音频文件
                if os.path.isdir(seq_path) and os.path.exists(audio_path):
                    seq_list.append(seq_dir)
                    seq_to_dir[seq_dir] = data_dir
                    dir_sequence_count += 1
            
            print(f"数据目录 {data_dir} 中找到 {dir_sequence_count} 个有效序列")
        else:
            missing = []
            if not os.path.exists(frames_dir):
                missing.append("align_frame目录")
            if not os.path.exists(audios_dir):
                missing.append("audios目录")
            print(f"警告：在 {data_dir} 中缺少 {', '.join(missing)}，跳过该数据集")
        
        return seq_list, seq_to_dir
    
    def __len__(self):
        return len(self.sequences) - len(self.blacklist_sequences)
    
    def read_image(self, image_path):
        """
        使用crop_face的裁剪逻辑读取图像：裁剪框底部与原图底部对齐，保留脖子
        使用瘦高比例：高度使用self.crop_scale，宽度使用self.crop_scale * 3/4
        """
        try:
            # 使用cv2直接读取
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
            
            # 获取原始图像尺寸
            height, width = img.shape[:2]
            
            # 计算裁剪尺寸 - 使用瘦高比例
            crop_height = int(min(width, height) * self.crop_scale)
            crop_width = int(min(width, height) * self.crop_scale * self.width_scale)  # 宽度是高度的3/4
            
            # 计算裁剪区域：底部对齐，水平居中
            x1 = width // 2 - crop_width // 2  # 水平居中
            y1 = height - crop_height  # 底部对齐
            
            # 确保坐标在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            
            # 裁剪图像
            cropped_img = img[y1:y1+crop_height, x1:x1+crop_width]
            
            # 调整为目标分辨率
            if cropped_img.shape[0] != self.resolution or cropped_img.shape[1] != self.resolution:
                resized_img = cv2.resize(cropped_img, (self.resolution, self.resolution))
            else:
                resized_img = cropped_img
            
            # 确保只有RGB通道
            if resized_img.shape[2] == 4:  # 如果有alpha通道
                resized_img = resized_img[:, :, :3]
                
            return resized_img # cv2.RGB
        except Exception as e:
            print(f"读取图像时出错: {e} - {image_path}")
            return None
    
    def read_audio_file(self, audio_path):
        """从音频文件读取音频并转换为 mel 频谱图"""
        try:
            # print(f"audio_path: {audio_path}")
            if os.path.exists(audio_path):
                ar = AudioReader(audio_path, ctx=cpu(self.worker_id), sample_rate=self.audio_sample_rate)
                original_mel = melspectrogram(ar[:].asnumpy().squeeze(0))
                return torch.from_numpy(original_mel)
            else:
                # print(f"音频文件不存在: {audio_path}")
                return None
        except Exception as e:
            # print(f"读取音频时出错: {e} - {audio_path}")
            return None
    
    def crop_audio_window(self, original_mel, start_frame):
        """从 mel 频谱图中裁剪出对应于特定起始帧的窗口"""
        start_idx = int(80.0 * (start_frame / float(self.video_fps)))
        end_idx = start_idx + self.mel_window_length
        
        # 确保索引有效
        if end_idx > original_mel.shape[1]:
            return None
            
        return original_mel[:, start_idx:end_idx].unsqueeze(0)
    
    def _cache_frame_paths(self, sequence_name):
        """缓存特定序列的帧路径"""
        # print(f"sequence_name: {sequence_name}")
        if sequence_name in self.frame_paths_cache:
            return self.frame_paths_cache[sequence_name]
            
        data_dir = self.sequence_to_data_dir[sequence_name]
        sequence_dir = os.path.join(data_dir, "align_frame", sequence_name)
        
        # 获取该序列中所有图像帧
        frame_paths = []
        try:
            for filename in sorted(os.listdir(sequence_dir)):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    frame_paths.append(os.path.join(sequence_dir, filename))
            
            # 按文件名排序（通常是数字顺序）
            frame_paths.sort()
            
            if len(frame_paths) < 2 * self.num_frames:
                # 帧数不足，加入黑名单
                # print(f"frame_paths has {len(frame_paths)} num_frames")
                return None
                
            # 缓存并返回
            self.frame_paths_cache[sequence_name] = frame_paths
            return frame_paths
        except Exception:
            return None
    
    def get_sequence(self, sequence_id, start_idx=None):
        """
        获取连续的图像序列和错误图像序列（用于创建不同步样本）
        
        Returns:
            tuple: (continuous_frames, wrong_frames, start_idx, audio_file)
                - continuous_frames: 连续帧序列
                - wrong_frames: 不同位置的帧序列
                - start_idx: 起始帧索引
                - audio_file: 对应的音频文件路径
        """
        # print(f"sequence_id: {sequence_id}")
        sequence_name = self.sequences[sequence_id]
        
        # 检查是否是已知的问题序列
        if sequence_name in self.blacklist_sequences:
            return None, None, None, None
            
        data_dir = self.sequence_to_data_dir[sequence_name]
        audio_file = os.path.join(data_dir, "audios", f"{sequence_name}.wav")
        
        # print(f"audio_file: {audio_file}")
        # 获取帧路径（优先使用缓存）
        frame_paths = self._cache_frame_paths(sequence_name)
        if frame_paths is None:
            self.blacklist_sequences.add(sequence_name)
            return None, None, None, None
        
        # print(f"frame_paths: {frame_paths}")
        total_num_frames = len(frame_paths)
        # print(f"total_num_frames: {total_num_frames}")
        # 如果未指定起始帧，随机选择一个起始帧
        if start_idx is None:
            max_start = total_num_frames - self.num_frames
            # print(f"max_start: {max_start}")
            if max_start < 0:
                self.blacklist_sequences.add(sequence_name)
                return None, None, None, None
            start_idx = random.randint(0, max_start)
        
        # print(f"start_idx: {start_idx}")

        # 获取连续帧
        continuous_frames = []
        for i in range(self.num_frames):
            if start_idx + i >= total_num_frames:
                self.blacklist_sequences.add(sequence_name)
                return None, None, None, None
            img = self.read_image(frame_paths[start_idx + i])
            # print(f"frame_paths[start_idx + i]: {frame_paths[start_idx + i]}")
            # print(f"img: {img}")
            if img is None:
                self.blacklist_sequences.add(sequence_name)
                return None, None, None, None
            continuous_frames.append(img)
        
        # print(f"continuous_frames: {continuous_frames}")
        # 获取错误帧（不同位置的帧）- 与原始SyncNet保持一致
        max_attempts = 10
        for _ in range(max_attempts):
            # 找一个与当前起始帧不同的位置
            wrong_start_idx = random.randint(0, total_num_frames - self.num_frames)
            # 确保不与原始序列重叠
            if wrong_start_idx == start_idx:
                continue
            # if wrong_start_idx >= start_idx - self.num_frames and wrong_start_idx <= start_idx + self.num_frames:
            #     continue
            wrong_frames = []
            valid_wrong = True
            for i in range(self.num_frames):
                if wrong_start_idx + i >= total_num_frames:
                    valid_wrong = False
                    break
                img = self.read_image(frame_paths[wrong_start_idx + i])
                if img is None:
                    valid_wrong = False
                    break
                wrong_frames.append(img)
            
            if valid_wrong:
                # print(f"return get_sequence")
                return np.array(continuous_frames), np.array(wrong_frames), start_idx, audio_file
        
        # 如果找不到有效的错误帧序列，返回None
        self.blacklist_sequences.add(sequence_name)
        return None, None, None, None

    def worker_init_fn(self, worker_id):
        """初始化工作进程"""
        self.worker_id = worker_id
        # 设置PyTorch的线程数，避免过度并行化
        torch.set_num_threads(1)
        
        # NumPy线程控制
        try:
            import numpy as np
            if hasattr(np, 'set_num_threads'):
                np.set_num_threads(1)
            elif hasattr(np, 'core') and hasattr(np.core, 'set_num_threads'):
                np.core.set_num_threads(1)
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
        except Exception as e:
            print(f"设置NumPy线程数时出错: {e}")
    
    def __getitem__(self, idx):
        """
        获取SyncNet训练样本，与原始SyncNetDataset保持一致的正负样本生成逻辑
        
        Returns:
            dict: 包含以下键的字典:
                - frames: 图像帧序列
                - audio_samples: 对应的mel频谱图
                - y: 同步标签（1为同步，0为不同步）
        """
        # print(f"idx: {idx}")
        max_attempts = 5
        
        for attempt in range(max_attempts):
            try:
            # 选择一个有效的序列ID
                valid_sequences = [i for i, s in enumerate(self.sequences) 
                                    if s not in self.blacklist_sequences]
                if not valid_sequences:
                    raise RuntimeError("没有有效的序列可用")
                
                sequence_id = random.choice(valid_sequences)
                
                # print(f"")
                # print(f"stesp 1: {sequence_id}")
                # 获取连续帧和错误帧
                continuous_frames, wrong_frames, start_idx, audio_file = self.get_sequence(sequence_id)
                # print(f"stesp 2: {continuous_frames.shape, wrong_frames.shape, start_idx, audio_file}")
                if continuous_frames is None or wrong_frames is None:
                    continue
                
        
                # 处理音频数据
                mel_cache_path = os.path.join(
                    self.audio_mel_cache_dir, 
                    os.path.basename(audio_file).replace('.wav', '_mel.pt')
                )
                # print(f"mel_cache_path: {mel_cache_path}")
                
                # 尝试加载缓存的mel文件
                if os.path.isfile(mel_cache_path):
                    try:
                        original_mel = torch.load(mel_cache_path)
                    except Exception as e:
                        os.remove(mel_cache_path)
                        original_mel = self.read_audio_file(audio_file)
                        if original_mel is not None:
                            torch.save(original_mel, mel_cache_path)
                        else:
                            continue
                else:
                    # print(f" new mel_cache_path: ")
                    original_mel = self.read_audio_file(audio_file)
                    # print(f" original_mel: {original_mel}")
                    if original_mel is not None:
                        torch.save(original_mel, mel_cache_path)
                    else:
                        continue
                
                # print(f"stesp 3: {original_mel}")

                # 裁剪mel到需要的窗口长度
                mel = self.crop_audio_window(original_mel, start_idx)
                if mel is None or mel.shape[-1] != self.mel_window_length:
                    continue
                
                # 随机选择使用正样本还是负样本 - 与原始SyncNetDataset保持一致
                if random.choice([True, False]):
                    # 同步样本 - 使用正确的连续帧
                    y = torch.ones(1).float()
                    chosen_frames = continuous_frames
                else:
                    # 不同步样本 - 使用错误帧
                    y = torch.zeros(1).float()
                    chosen_frames = wrong_frames
                
                # print(f"stesp 4: {chosen_frames, y}")
                # 使用image_processor处理选中的帧
                processed_frames = self.image_processor.process_images(chosen_frames)
                
                # 返回样本 - 与原始格式保持一致
                sample = {
                    'frames': processed_frames,  # 已处理的帧序列
                    'audio_samples': mel,        # mel频谱图
                    'y': y                       # 同步标签
                }
                
                return sample
                    
            except Exception as e:
                print(f"获取样本时出错: {e}")
                continue
                
        # 如果所有尝试都失败，返回随机样本
        return None
        # return self.__getitem__(random.randint(0, len(self) - 1))





class SyncNetDataset_official_processor_with_our_data_v2(Dataset):
    """
    SyncNet数据集 - 基于SyncNet模型需求设计，同时适配自定义数据处理逻辑
    与原始SyncNetDataset保持一致的参数设置方式
    """
    def __init__(self, train_data_dir: list, config):
        """
        初始化SyncNet数据集
        
        Args:
            train_data_dir: 训练数据根目录列表
            config: 配置对象
        """
        if isinstance(train_data_dir, str):
            train_data_dir = [train_data_dir]  # 兼容单目录情况
        
        self.train_data_dirs = train_data_dir
        self.sequences = []
        self.sequence_to_data_dir = {}  # 记录每个序列对应的数据目录
        
        # 使用线程池并行扫描所有数据目录
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(self.train_data_dirs), 8)) as executor:
            results = list(executor.map(self._scan_data_dir, self.train_data_dirs))
        
        # 合并结果
        for seq_list, seq_to_dir in results:
            self.sequences.extend(seq_list)
            self.sequence_to_data_dir.update(seq_to_dir)
        
        if len(self.sequences) == 0:
            raise ValueError("未找到有效的序列列表")
        
        # 基本参数设置 - 与原始SyncNetDataset保持一致
        self.resolution = config.data.resolution
        self.num_frames = config.data.num_frames
        
        # mel频谱图窗口长度计算方式与原始一致
        self.mel_window_length = math.ceil(self.num_frames / 5 * 16)
        
        self.audio_sample_rate = config.data.audio_sample_rate
        self.video_fps = config.data.video_fps
        self.audio_samples_length = int(
            config.data.audio_sample_rate // config.data.video_fps * config.data.num_frames
        )
        
        # 图像处理器设置与原始一致
        # self.image_processor = ImageProcessor(resolution=config.data.resolution, mask="mouth")
        self.image_processor = ImageProcessor(resolution=config.data.resolution, mask="fix_mask", device="cuda")
        # self.image_processor = ImageProcessor(resolution=config.data.resolution, mask="fix_mask")
        
        # 音频处理相关
        self.audio_mel_cache_dir = config.data.audio_mel_cache_dir
        os.makedirs(self.audio_mel_cache_dir, exist_ok=True)
        
        # # 特有设置
        # self.mask_path = "/home/work/houyi/pj_25_q1/LatentSync/hy_test/crop_face/narrowed_mask/output_mask_09.png"
        # self.mask_image = load_fixed_mask(self.resolution, mask_path=self.mask_path)
        
        # 裁剪比例，与crop_face_no_margin_video_v2.py保持一致
        self.crop_scale = 0.78125  # 默认值 200/256
        # self.crop_scale = 1 
        self.width_scale = 1.
        # self.width_scale = 1.
    
        self.ratio = 2.8
        self.crop_ratio = (self.ratio, self.ratio)

        self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
        self.border_mode = cv2.BORDER_CONSTANT

        # 黑名单和缓存
        self.frame_paths_cache = {}
        self.blacklist_sequences = set()  # 存储问题序列
        
        print(f"初始化SyncNetDataset_v1, 总共找到 {len(self.sequences)} 个序列")
        print(f"帧数: {self.num_frames}, 分辨率: {self.resolution}, mel长度: {self.mel_window_length}")
        print(f"使用裁剪比例: {self.crop_scale}")
        # print(f"使用mask路径: {self.mask_path}")
    
    def _scan_data_dir(self, data_dir):
        """
        扫描单个数据目录并返回序列列表
        """
        seq_list = []
        seq_to_dir = {}
        
        frames_dir = os.path.join(data_dir, "frames")
        audios_dir = os.path.join(data_dir, "audios")
        
        dir_sequence_count = 0  # 记录当前数据目录中找到的序列数量
        
        if os.path.exists(frames_dir) and os.path.exists(audios_dir):
            for seq_dir in os.listdir(frames_dir):
                seq_path = os.path.join(frames_dir, seq_dir)
                audio_path = os.path.join(audios_dir, f"{seq_dir}.wav")
                
                # 检查是否同时存在帧目录和对应的音频文件
                if os.path.isdir(seq_path) and os.path.exists(audio_path):
                    seq_list.append(seq_dir)
                    seq_to_dir[seq_dir] = data_dir
                    dir_sequence_count += 1
            
            print(f"数据目录 {data_dir} 中找到 {dir_sequence_count} 个有效序列")
        else:
            missing = []
            if not os.path.exists(frames_dir):
                missing.append("frames目录")
            if not os.path.exists(audios_dir):
                missing.append("audios目录")
            print(f"警告：在 {data_dir} 中缺少 {', '.join(missing)}，跳过该数据集")
        
        return seq_list, seq_to_dir
    
    def __len__(self):
        return len(self.sequences) - len(self.blacklist_sequences)
    
    # def read_image(self, image_path):
    #     """
    #     使用crop_face的裁剪逻辑读取图像：裁剪框底部与原图底部对齐，保留脖子
    #     使用瘦高比例：高度使用self.crop_scale，宽度使用self.crop_scale * 3/4
    #     """
    #     try:
    #         # 使用cv2直接读取
    #         img = cv2.imread(image_path)
    #         if img is None:
    #             raise ValueError(f"无法读取图像: {image_path}")
                
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
            
    #         # 获取原始图像尺寸
    #         height, width = img.shape[:2]
            
    #         # 计算裁剪尺寸 - 使用瘦高比例
    #         crop_height = int(min(width, height) * self.crop_scale)
    #         crop_width = int(min(width, height) * self.crop_scale * self.width_scale)  # 宽度是高度的3/4
            
    #         # 计算裁剪区域：底部对齐，水平居中
    #         x1 = width // 2 - crop_width // 2  # 水平居中
    #         y1 = height - crop_height  # 底部对齐
            
    #         # 确保坐标在图像范围内
    #         x1 = max(0, x1)
    #         y1 = max(0, y1)
            
    #         # 裁剪图像
    #         cropped_img = img[y1:y1+crop_height, x1:x1+crop_width]
            
    #         # 调整为目标分辨率
    #         if cropped_img.shape[0] != self.resolution or cropped_img.shape[1] != self.resolution:
    #             resized_img = cv2.resize(cropped_img, (self.resolution, self.resolution))
    #         else:
    #             resized_img = cropped_img
            
    #         # 确保只有RGB通道
    #         if resized_img.shape[2] == 4:  # 如果有alpha通道
    #             resized_img = resized_img[:, :, :3]
                
    #         return resized_img
    #     except Exception as e:
    #         print(f"读取图像时出错: {e} - {image_path}")
    #         return None


    def read_image(self, image_path):
        """
        使用crop_face的裁剪逻辑读取图像：裁剪框底部与原图底部对齐，保留脖子
        使用瘦高比例：高度使用self.crop_scale，宽度使用self.crop_scale * 3/4
        """
        # try:
        # 使用cv2直接读取
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB

        # ratio = 2.8
        # crop_ratio = (ratio, ratio)

        # face_size = (int(75 * crop_ratio[0]), int(100 * crop_ratio[1]))
        # print(f"face_size is : {face_size}")

        # border_mode = cv2.BORDER_CONSTANT
        face, box, affine_matrix = self.image_processor.affine_transform(frame_rgb)
        resized_img = rearrange(face, "c h w -> h w c").numpy()
        # cropped_face = cv2.warpAffine(
        #             frame_rgb,
        #             affine_matrix,
        #             self.face_size,
        #             flags=cv2.INTER_LANCZOS4,
        #             borderMode=self.border_mode,
        #             borderValue=[127, 127, 127],
        #         )


        # # 获取原始图像尺寸
        # height, width = img.shape[:2]
        
        # # 计算裁剪尺寸 - 使用瘦高比例
        # crop_height = int(min(width, height) * self.crop_scale)
        # crop_width = int(min(width, height) * self.crop_scale * self.width_scale)  # 宽度是高度的3/4
        
        # # 计算裁剪区域：底部对齐，水平居中
        # x1 = width // 2 - crop_width // 2  # 水平居中
        # y1 = height - crop_height  # 底部对齐
        
        # # 确保坐标在图像范围内
        # x1 = max(0, x1)
        # y1 = max(0, y1)
        
        # # 裁剪图像
        # cropped_img = img[y1:y1+crop_height, x1:x1+crop_width]
        
        # # 调整为目标分辨率
        # if cropped_img.shape[0] != self.resolution or cropped_img.shape[1] != self.resolution:
        #     resized_img = cv2.resize(cropped_img, (self.resolution, self.resolution))
        # else:
        #     resized_img = cropped_img
        
        # # 确保只有RGB通道
        # if resized_img.shape[2] == 4:  # 如果有alpha通道
        #     resized_img = resized_img[:, :, :3]
            
        return resized_img
        # except Exception as e:
        #     print(f"读取图像时出错: {e} - {image_path}")
        #     return None

    def read_audio_file(self, audio_path):
        """从音频文件读取音频并转换为 mel 频谱图"""
        try:
            # print(f"audio_path: {audio_path}")
            if os.path.exists(audio_path):
                ar = AudioReader(audio_path, ctx=cpu(self.worker_id), sample_rate=self.audio_sample_rate)
                original_mel = melspectrogram(ar[:].asnumpy().squeeze(0))
                return torch.from_numpy(original_mel)
            else:
                # print(f"音频文件不存在: {audio_path}")
                return None
        except Exception as e:
            # print(f"读取音频时出错: {e} - {audio_path}")
            return None
    
    def crop_audio_window(self, original_mel, start_frame):
        """从 mel 频谱图中裁剪出对应于特定起始帧的窗口"""
        start_idx = int(80.0 * (start_frame / float(self.video_fps)))
        end_idx = start_idx + self.mel_window_length
        
        # 确保索引有效
        if end_idx > original_mel.shape[1]:
            return None
            
        return original_mel[:, start_idx:end_idx].unsqueeze(0)
    
    def _cache_frame_paths(self, sequence_name):
        """缓存特定序列的帧路径"""
        # print(f"sequence_name: {sequence_name}")
        if sequence_name in self.frame_paths_cache:
            return self.frame_paths_cache[sequence_name]
            
        data_dir = self.sequence_to_data_dir[sequence_name]
        sequence_dir = os.path.join(data_dir, "frames", sequence_name)
        
        # 获取该序列中所有图像帧
        frame_paths = []
        try:
            for filename in sorted(os.listdir(sequence_dir)):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    frame_paths.append(os.path.join(sequence_dir, filename))
            
            # 按文件名排序（通常是数字顺序）
            frame_paths.sort()
            
            if len(frame_paths) < 2 * self.num_frames:
                # 帧数不足，加入黑名单
                # print(f"frame_paths has {len(frame_paths)} num_frames")
                return None
                
            # 缓存并返回
            self.frame_paths_cache[sequence_name] = frame_paths
            return frame_paths
        except Exception:
            return None
    
    def get_sequence(self, sequence_id, start_idx=None):
        """
        获取连续的图像序列和错误图像序列（用于创建不同步样本）
        
        Returns:
            tuple: (continuous_frames, wrong_frames, start_idx, audio_file)
                - continuous_frames: 连续帧序列
                - wrong_frames: 不同位置的帧序列
                - start_idx: 起始帧索引
                - audio_file: 对应的音频文件路径
        """
        # print(f"sequence_id: {sequence_id}")
        sequence_name = self.sequences[sequence_id]
        
        # 检查是否是已知的问题序列
        if sequence_name in self.blacklist_sequences:
            return None, None, None, None
            
        data_dir = self.sequence_to_data_dir[sequence_name]
        audio_file = os.path.join(data_dir, "audios", f"{sequence_name}.wav")
        
        # print(f"audio_file: {audio_file}")
        # 获取帧路径（优先使用缓存）
        frame_paths = self._cache_frame_paths(sequence_name)
        if frame_paths is None:
            self.blacklist_sequences.add(sequence_name)
            return None, None, None, None
        
        # print(f"frame_paths: {frame_paths}")
        total_num_frames = len(frame_paths)
        # print(f"total_num_frames: {total_num_frames}")
        # 如果未指定起始帧，随机选择一个起始帧
        if start_idx is None:
            max_start = total_num_frames - self.num_frames
            # print(f"max_start: {max_start}")
            if max_start < 0:
                self.blacklist_sequences.add(sequence_name)
                return None, None, None, None
            start_idx = random.randint(0, max_start)
        
        # print(f"start_idx: {start_idx}")

        # 获取连续帧
        continuous_frames = []
        for i in range(self.num_frames):
            if start_idx + i >= total_num_frames:
                self.blacklist_sequences.add(sequence_name)
                return None, None, None, None
            img = self.read_image(frame_paths[start_idx + i])
            # print(f"frame_paths[start_idx + i]: {frame_paths[start_idx + i]}")
            # print(f"img: {img}")
            if img is None:
                self.blacklist_sequences.add(sequence_name)
                return None, None, None, None
            continuous_frames.append(img)
        
        # print(f"continuous_frames: {continuous_frames}")
        # 获取错误帧（不同位置的帧）- 与原始SyncNet保持一致
        max_attempts = 10
        for _ in range(max_attempts):
            # 找一个与当前起始帧不同的位置
            wrong_start_idx = random.randint(0, total_num_frames - self.num_frames)
            # 确保不与原始序列重叠
            if wrong_start_idx == start_idx:
                continue
            # if wrong_start_idx >= start_idx - self.num_frames and wrong_start_idx <= start_idx + self.num_frames:
            #     continue
            wrong_frames = []
            valid_wrong = True
            for i in range(self.num_frames):
                if wrong_start_idx + i >= total_num_frames:
                    valid_wrong = False
                    break
                img = self.read_image(frame_paths[wrong_start_idx + i])
                if img is None:
                    valid_wrong = False
                    break
                wrong_frames.append(img)
            
            if valid_wrong:
                # print(f"return get_sequence")
                return np.array(continuous_frames), np.array(wrong_frames), start_idx, audio_file
        
        # 如果找不到有效的错误帧序列，返回None
        self.blacklist_sequences.add(sequence_name)
        return None, None, None, None

    def worker_init_fn(self, worker_id):
        """初始化工作进程"""
        self.worker_id = worker_id
        # 设置PyTorch的线程数，避免过度并行化
        torch.set_num_threads(1)
        
        # NumPy线程控制
        try:
            import numpy as np
            if hasattr(np, 'set_num_threads'):
                np.set_num_threads(1)
            elif hasattr(np, 'core') and hasattr(np.core, 'set_num_threads'):
                np.core.set_num_threads(1)
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
        except Exception as e:
            print(f"设置NumPy线程数时出错: {e}")
    
    def __getitem__(self, idx):
        """
        获取SyncNet训练样本，与原始SyncNetDataset保持一致的正负样本生成逻辑
        
        Returns:
            dict: 包含以下键的字典:
                - frames: 图像帧序列
                - audio_samples: 对应的mel频谱图
                - y: 同步标签（1为同步，0为不同步）
        """
        # print(f"idx: {idx}")
        max_attempts = 5
        
        for attempt in range(max_attempts):
            try:
            # 选择一个有效的序列ID
                valid_sequences = [i for i, s in enumerate(self.sequences) 
                                    if s not in self.blacklist_sequences]
                if not valid_sequences:
                    raise RuntimeError("没有有效的序列可用")
                
                sequence_id = random.choice(valid_sequences)
                
                # print(f"")
                # print(f"stesp 1: {sequence_id}")
                # 获取连续帧和错误帧
                continuous_frames, wrong_frames, start_idx, audio_file = self.get_sequence(sequence_id)
                # print(f"stesp 2: {continuous_frames.shape, wrong_frames.shape, start_idx, audio_file}")
                if continuous_frames is None or wrong_frames is None:
                    continue
                
        
                # 处理音频数据
                mel_cache_path = os.path.join(
                    self.audio_mel_cache_dir, 
                    os.path.basename(audio_file).replace('.wav', '_mel.pt')
                )
                # print(f"mel_cache_path: {mel_cache_path}")
                
                # 尝试加载缓存的mel文件
                if os.path.isfile(mel_cache_path):
                    try:
                        original_mel = torch.load(mel_cache_path)
                    except Exception as e:
                        os.remove(mel_cache_path)
                        original_mel = self.read_audio_file(audio_file)
                        if original_mel is not None:
                            torch.save(original_mel, mel_cache_path)
                        else:
                            continue
                else:
                    # print(f" new mel_cache_path: ")
                    original_mel = self.read_audio_file(audio_file)
                    # print(f" original_mel: {original_mel}")
                    if original_mel is not None:
                        torch.save(original_mel, mel_cache_path)
                    else:
                        continue
                
                # print(f"stesp 3: {original_mel}")

                # 裁剪mel到需要的窗口长度
                mel = self.crop_audio_window(original_mel, start_idx)
                if mel is None or mel.shape[-1] != self.mel_window_length:
                    continue
                
                # 随机选择使用正样本还是负样本 - 与原始SyncNetDataset保持一致
                if random.choice([True, False]):
                    # 同步样本 - 使用正确的连续帧
                    y = torch.ones(1).float()
                    chosen_frames = continuous_frames
                else:
                    # 不同步样本 - 使用错误帧
                    y = torch.zeros(1).float()
                    chosen_frames = wrong_frames
                
                # print(f"stesp 4: {chosen_frames, y}")
                # 使用image_processor处理选中的帧
                processed_frames = self.image_processor.process_images(chosen_frames)
                
                # 返回样本 - 与原始格式保持一致
                sample = {
                    'frames': processed_frames,  # 已处理的帧序列
                    'audio_samples': mel,        # mel频谱图
                    'y': y                       # 同步标签
                }
                
                return sample
                    
            except Exception as e:
                print(f"获取样本时出错: {e}")
                continue
                
        # 如果所有尝试都失败，返回随机样本
        # return None
        return self.__getitem__(random.randint(0, len(self) - 1))


#   File "/home/work/houyi/pj_25_q1/LatentSync/latentsync/data/syncnet_dataset.py", line 910, in get_sequence
#     img = self.read_image(frame_paths[start_idx + i])
#   File "/home/work/houyi/pj_25_q1/LatentSync/latentsync/data/syncnet_dataset.py", line 761, in read_image
#     face, box, affine_matrix = self.image_processor.affine_transform(frame_rgb)
#   File "/home/work/houyi/pj_25_q1/LatentSync/latentsync/utils/image_processor.py", line 127, in affine_transform
#     raise RuntimeError("Face not detected")