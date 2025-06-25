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
import cv2
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..utils.audio import melspectrogram
from decord import AudioReader, VideoReader, cpu


class UNetDataset(Dataset):
    def __init__(self, train_data_dir: str, config):
        if config.data.train_fileslist != "":
            with open(config.data.train_fileslist) as file:
                self.video_paths = [line.rstrip() for line in file]
        elif train_data_dir != "":
            self.video_paths = []
            for file in os.listdir(train_data_dir):
                if file.endswith(".mp4"):
                    self.video_paths.append(os.path.join(train_data_dir, file))
        else:
            raise ValueError("data_dir and fileslist cannot be both empty")

        self.resolution = config.data.resolution
        self.num_frames = config.data.num_frames

        if self.num_frames == 16:
            self.mel_window_length = 52
        elif self.num_frames == 5:
            self.mel_window_length = 16
        else:
            raise NotImplementedError("Only support 16 and 5 frames now")

        self.audio_sample_rate = config.data.audio_sample_rate
        self.video_fps = config.data.video_fps
        self.mask = config.data.mask
        self.mask_image = load_fixed_mask(self.resolution)
        self.load_audio_data = config.model.add_audio_layer and config.run.use_syncnet
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

        start_idx = random.randint(self.num_frames // 2, total_num_frames - self.num_frames - self.num_frames // 2)
        frames_index = np.arange(start_idx, start_idx + self.num_frames, dtype=int)

        while True:
            wrong_start_idx = random.randint(0, total_num_frames - self.num_frames)
            if wrong_start_idx > start_idx - self.num_frames and wrong_start_idx < start_idx + self.num_frames:
                continue
            wrong_frames_index = np.arange(wrong_start_idx, wrong_start_idx + self.num_frames, dtype=int)
            break

        frames = video_reader.get_batch(frames_index).asnumpy()
        wrong_frames = video_reader.get_batch(wrong_frames_index).asnumpy()

        return frames, wrong_frames, start_idx

    def worker_init_fn(self, worker_id):
        """初始化工作进程"""
        self.worker_id = worker_id
        # 设置PyTorch的线程数，避免过度并行化
        torch.set_num_threads(1)
        
        # NumPy线程控制 - 兼容不同版本的NumPy
        try:
            import numpy as np
            # 尝试不同的NumPy线程控制方法
            if hasattr(np, 'set_num_threads'):
                np.set_num_threads(1)
            elif hasattr(np, 'core') and hasattr(np.core, 'set_num_threads'):
                np.core.set_num_threads(1)  # 有些版本在np.core中
            # 环境变量方式控制线程数
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
        except Exception as e:
            print(f"设置NumPy线程数时出错: {e}")
        
        setattr(
            self,
            f"image_processor_{worker_id}",
            ImageProcessor(self.resolution, self.mask, mask_image=self.mask_image),
        )

    def __getitem__(self, idx):
        image_processor = getattr(self, f"image_processor_{self.worker_id}")
        while True:
            try:
                idx = random.randint(0, len(self) - 1)

                # Get video file path
                video_path = self.video_paths[idx]

                vr = VideoReader(video_path, ctx=cpu(self.worker_id))

                if len(vr) < 3 * self.num_frames:
                    continue

                continuous_frames, ref_frames, start_idx = self.get_frames(vr)

                if self.load_audio_data:
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
                else:
                    mel = []

                gt, masked_gt, mask = image_processor.prepare_masks_and_masked_images(
                    continuous_frames, affine_transform=False
                )

                if self.mask == "fix_mask":
                    ref, _, _ = image_processor.prepare_masks_and_masked_images(ref_frames, affine_transform=False)
                else:
                    ref = image_processor.process_images(ref_frames)
                vr.seek(0)  # avoid memory leak
                break

            except Exception as e:  # Handle the exception of face not detcted
                print(f"{type(e).__name__} - {e} - {video_path}")
                if "vr" in locals():
                    vr.seek(0)  # avoid memory leak

        sample = dict(
            gt=gt,
            masked_gt=masked_gt,
            ref=ref,
            mel=mel,
            mask=mask,
            video_path=video_path,
            start_idx=start_idx,
        )

        return sample


from PIL import Image

class CustomUNetDataset_384_hy_0227_v1(Dataset):
    def __init__(self, train_data_dir: str, config):
        """
        初始化自定义数据集，适应用户特定的数据格式
        
        Args:
            train_data_dir: 训练数据根目录，包含align_frame和audios子目录
            config: 配置对象
        """
        self.train_data_dir = train_data_dir
        
        # 加载序列列表
        if config.data.train_fileslist != "":
            with open(config.data.train_fileslist) as file:
                self.sequences = [line.rstrip() for line in file]
        else:
            # 如果未提供列表文件，则尝试从namelist.txt加载
            namelist_path = os.path.join(train_data_dir, "namelist.txt")
            if os.path.exists(namelist_path):
                with open(namelist_path) as file:
                    self.sequences = [line.rstrip() for line in file]
            else:
                # 如果namelist.txt也不存在，则尝试直接从目录结构获取序列列表
                self.sequences = []
                frames_dir = os.path.join(train_data_dir, "align_frame")
                if os.path.exists(frames_dir):
                    for seq_dir in os.listdir(frames_dir):
                        if os.path.isdir(os.path.join(frames_dir, seq_dir)):
                            self.sequences.append(seq_dir)
                else:
                    raise ValueError(f"无法找到帧目录：{frames_dir}")
        
        if len(self.sequences) == 0:
            raise ValueError("未找到有效的序列列表")
            
        # 图像和音频目录
        self.frames_dir = os.path.join(train_data_dir, "align_frame")
        self.audio_dir = os.path.join(train_data_dir, "audios")
        
        self.resolution = config.data.resolution  # 应为384
        self.num_frames = config.data.num_frames
        
        if self.num_frames == 16:
            self.mel_window_length = 52
        elif self.num_frames == 5:
            self.mel_window_length = 16
        else:
            raise NotImplementedError("目前仅支持16帧和5帧")
        
        self.audio_sample_rate = config.data.audio_sample_rate
        self.video_fps = config.data.video_fps
        self.mask = config.data.mask
        self.mask_image = load_fixed_mask(self.resolution)
        self.load_audio_data = config.model.add_audio_layer and config.run.use_syncnet
        self.audio_mel_cache_dir = config.data.audio_mel_cache_dir
        os.makedirs(self.audio_mel_cache_dir, exist_ok=True)
        
        # 验证数据集结构
        print(f"找到 {len(self.sequences)} 个序列")
        
    def __len__(self):
        return len(self.sequences)
    
    def read_image(self, image_path):
        """读取图像并转换为适当的格式，调整为384x384分辨率"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = np.array(img)
            # 确保图像大小为384x384
            if img.shape[0] != self.resolution or img.shape[1] != self.resolution:
                img = cv2.resize(img, (self.resolution, self.resolution))
            # 转换为 RGB 格式 (H, W, C)
            if img.shape[2] == 4:  # 如果有alpha通道
                img = img[:, :, :3]
            return img
        except Exception as e:
            print(f"读取图像时出错: {e} - {image_path}")
            return None
    
    def read_audio_file(self, audio_path):
        # print(f"audio_path: {audio_path}")
        """从音频文件读取音频并转换为 mel 频谱图"""
        try:
            if os.path.exists(audio_path):
                ar = AudioReader(audio_path, ctx=cpu(self.worker_id), sample_rate=self.audio_sample_rate)
                original_mel = melspectrogram(ar[:].asnumpy().squeeze(0))
                return torch.from_numpy(original_mel)
            else:
                print(f"音频文件不存在: {audio_path}")
                return None
        except Exception as e:
            print(f"读取音频时出错: {e} - {audio_path}")
            return None
    
    def crop_audio_window(self, original_mel, start_frame):
        """从 mel 频谱图中裁剪出对应于特定起始帧的窗口"""
        start_idx = int(80.0 * (start_frame / float(self.video_fps)))
        end_idx = start_idx + self.mel_window_length
        
        # 确保索引有效
        if end_idx > original_mel.shape[1]:
            return None
            
        return original_mel[:, start_idx:end_idx].unsqueeze(0)
    
    def get_sequence(self, sequence_id, start_idx=None):
        """获取连续的图像序列和参考图像序列"""
        sequence_name = self.sequences[sequence_id]
        sequence_dir = os.path.join(self.frames_dir, sequence_name)
        audio_file = os.path.join(self.audio_dir, f"{sequence_name}.wav")
        
        # 获取该序列中所有图像帧
        frame_paths = []
        for filename in sorted(os.listdir(sequence_dir)):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                frame_paths.append(os.path.join(sequence_dir, filename))
        
        # 按文件名排序（通常是数字顺序）
        frame_paths.sort()
        
        # 如果帧数不够，返回None
        if len(frame_paths) < 2 * self.num_frames:
            print(f"序列 {sequence_name} 的帧数不足 ({len(frame_paths)})")
            return None, None, None, audio_file
        
        # 如果未指定起始帧，随机选择一个起始帧
        if start_idx is None:
            max_start = len(frame_paths) - 2 * self.num_frames
            if max_start < 0:
                print(f"序列 {sequence_name} 帧数不足")
                return None, None, None, audio_file
            start_idx = random.randint(0, max_start)
        
        # 获取连续帧
        continuous_frames = []
        for i in range(self.num_frames):
            if start_idx + i >= len(frame_paths):
                print(f"序列 {sequence_name} 的帧索引越界: {start_idx + i} >= {len(frame_paths)}")
                return None, None, None, audio_file
            img = self.read_image(frame_paths[start_idx + i])
            if img is None:
                return None, None, None, audio_file
            continuous_frames.append(img)
        
        # 获取参考帧（从不同位置选择）
        max_attempts = 10
        for _ in range(max_attempts):
            if len(frame_paths) <= self.num_frames:
                print(f"序列 {sequence_name} 帧数不足以选择参考帧")
                return None, None, None, audio_file
                
            wrong_start_idx = random.randint(0, len(frame_paths) - self.num_frames)
            # 确保参考帧与连续帧不重叠
            if wrong_start_idx > start_idx - self.num_frames and wrong_start_idx < start_idx + self.num_frames:
                continue
                
            ref_frames = []
            valid_ref = True
            for i in range(self.num_frames):
                if wrong_start_idx + i >= len(frame_paths):
                    valid_ref = False
                    break
                img = self.read_image(frame_paths[wrong_start_idx + i])
                if img is None:
                    valid_ref = False
                    break
                ref_frames.append(img)
            
            if valid_ref:
                return np.array(continuous_frames), np.array(ref_frames), start_idx, audio_file
        
        print(f"无法为序列 {sequence_name} 找到有效的参考帧")
        return None, None, None, audio_file
    
    def worker_init_fn(self, worker_id):
        """初始化工作进程"""
        self.worker_id = worker_id
        setattr(
            self,
            f"image_processor_{worker_id}",
            ImageProcessor(self.resolution, self.mask, mask_image=self.mask_image),
        )
    
    def __getitem__(self, idx):
        """获取数据样本"""
        image_processor = getattr(self, f"image_processor_{self.worker_id}")
        max_attempts = 10
        
        for attempt in range(max_attempts):
            try:
                # 随机选择一个序列
                sequence_id = random.randint(0, len(self) - 1)
                
                # 获取连续帧和参考帧
                continuous_frames, ref_frames, start_idx, audio_file = self.get_sequence(sequence_id)
                if continuous_frames is None or ref_frames is None:
                    continue
                
                # 处理音频数据
                if self.load_audio_data:
                    mel_cache_path = os.path.join(
                        self.audio_mel_cache_dir, os.path.basename(audio_file) + "_mel.pt"
                    )
                    
                    if os.path.isfile(mel_cache_path):
                        try:
                            original_mel = torch.load(mel_cache_path)
                        except Exception as e:
                            print(f"{type(e).__name__} - {e} - {mel_cache_path}")
                            os.remove(mel_cache_path)
                            original_mel = self.read_audio_file(audio_file)
                            if original_mel is not None:
                                torch.save(original_mel, mel_cache_path)
                            else:
                                continue
                    else:
                        original_mel = self.read_audio_file(audio_file)
                        if original_mel is not None:
                            torch.save(original_mel, mel_cache_path)
                        else:
                            continue
                    
                    mel = self.crop_audio_window(original_mel, start_idx)
                    if mel is None or mel.shape[-1] != self.mel_window_length:
                        continue
                else:
                    mel = []
                
                # 使用图像处理器准备遮罩和遮罩图像
                gt, masked_gt, mask = image_processor.prepare_masks_and_masked_images(
                    continuous_frames, affine_transform=False
                )
                
                if self.mask == "fix_mask":
                    ref, _, _ = image_processor.prepare_masks_and_masked_images(ref_frames, affine_transform=False)
                else:
                    ref = image_processor.process_images(ref_frames)
                
                break
            
            except Exception as e:
                print(f"{type(e).__name__} - {e} - 序列ID: {sequence_id}, 尝试: {attempt+1}/{max_attempts}")
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"在 {max_attempts} 次尝试后无法加载有效样本")
                continue
        
        # 返回样本字典
        sample = dict(
            gt=gt,
            masked_gt=masked_gt,
            ref=ref,
            mel=mel,
            mask=mask,
            video_path=audio_file,  # 用序列名称替代视频路径
            start_idx=start_idx,
        )
        
        return sample
    


class CustomUNetDataset_384_hy_0227_v2(Dataset):
    def __init__(self, train_data_dir: list, config):
        """
        初始化自定义数据集，适应用户特定的数据格式，支持多个数据目录
        
        Args:
            train_data_dir: 训练数据根目录列表，每个目录包含align_frame和audios子目录
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
            
        self.resolution = config.data.resolution  # 应为384
        self.num_frames = config.data.num_frames
        
        if self.num_frames == 16:
            self.mel_window_length = 52
        elif self.num_frames == 5:
            self.mel_window_length = 16
        else:
            raise NotImplementedError("目前仅支持16帧和5帧")
        
        self.audio_sample_rate = config.data.audio_sample_rate
        self.video_fps = config.data.video_fps
        self.mask = config.data.mask
        self.mask_image = load_fixed_mask(self.resolution)
        self.load_audio_data = config.model.add_audio_layer and config.run.use_syncnet
        self.audio_mel_cache_dir = config.data.audio_mel_cache_dir
        os.makedirs(self.audio_mel_cache_dir, exist_ok=True)
        
        # 预缓存有效帧路径
        self.frame_paths_cache = {}
        self.blacklist_sequences = set()  # 存储问题序列
        
        # 预缓存一部分音频mel谱图
        if self.load_audio_data:
            self._preload_audio_mels(config.data.preload_mels_fraction)
        
        # 验证数据集结构
        print(f"总共找到 {len(self.sequences)} 个序列，来自 {len(self.train_data_dirs)} 个数据目录")
    
    def _scan_data_dir(self, data_dir):
        """
        扫描单个数据目录并返回序列列表
        
        Args:
            data_dir: 数据目录路径
            
        Returns:
            (seq_list, seq_to_dir): 序列列表和序列到目录的映射
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
    
    def _preload_audio_mels(self, preload_fraction=0.1):
        """预加载部分音频的mel谱图到内存中"""
        if preload_fraction <= 0:
            return
            
        num_to_preload = int(len(self.sequences) * preload_fraction)
        if num_to_preload <= 0:
            return
            
        print(f"预加载 {num_to_preload} 个音频的mel谱图...")
        import random
        sequences_to_preload = random.sample(self.sequences, num_to_preload)
        
        # 使用进程池并行预加载
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
            # 定义一个加载单个mel的函数
            def load_mel(seq_name):
                data_dir = self.sequence_to_data_dir[seq_name]
                audio_file = os.path.join(data_dir, "audios", f"{seq_name}.wav")
                mel_cache_path = os.path.join(self.audio_mel_cache_dir, f"{seq_name}_mel.pt")
                
                if not os.path.isfile(mel_cache_path):
                    try:
                        # 在这里我们需要一个不依赖self.worker_id的AudioReader版本
                        import librosa
                        y, sr = librosa.load(audio_file, sr=self.audio_sample_rate)
                        # 简化的melspectrogram计算，您需要根据实际情况调整参数
                        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
                        original_mel = torch.from_numpy(S)
                        torch.save(original_mel, mel_cache_path)
                        return seq_name, True
                    except Exception:
                        return seq_name, False
                return seq_name, True
                
            # 并行加载
            for seq_name, success in executor.map(load_mel, sequences_to_preload):
                if not success:
                    print(f"预加载 {seq_name} 的mel谱图失败")
    
    def __len__(self):
        return len(self.sequences) - len(self.blacklist_sequences)
    
    def read_image(self, image_path):
        """读取图像并转换为适当的格式，调整为384x384分辨率"""
        try:
            # 使用cv2直接读取更快
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
            
            # 确保图像大小为384x384
            if img.shape[0] != self.resolution or img.shape[1] != self.resolution:
                img = cv2.resize(img, (self.resolution, self.resolution))
            
            # 转换为 RGB 格式 (H, W, C)
            if img.shape[2] == 4:  # 如果有alpha通道
                img = img[:, :, :3]
                
            return img
        except Exception as e:
            # print(f"读取图像时出错: {e} - {image_path}")
            return None
    
    def read_audio_file(self, audio_path):
        """从音频文件读取音频并转换为 mel 频谱图"""
        try:
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
                return None
                
            # 缓存并返回
            self.frame_paths_cache[sequence_name] = frame_paths
            return frame_paths
        except Exception:
            return None
    
    def get_sequence(self, sequence_id, start_idx=None):
        """获取连续的图像序列和参考图像序列"""
        sequence_name = self.sequences[sequence_id]
        
        # 检查是否是已知的问题序列
        if sequence_name in self.blacklist_sequences:
            return None, None, None, None
            
        data_dir = self.sequence_to_data_dir[sequence_name]
        audio_file = os.path.join(data_dir, "audios", f"{sequence_name}.wav")
        
        # 获取帧路径（优先使用缓存）
        frame_paths = self._cache_frame_paths(sequence_name)
        if frame_paths is None:
            self.blacklist_sequences.add(sequence_name)
            return None, None, None, audio_file
        
        # 如果未指定起始帧，随机选择一个起始帧
        if start_idx is None:
            max_start = len(frame_paths) - 2 * self.num_frames
            if max_start < 0:
                self.blacklist_sequences.add(sequence_name)
                return None, None, None, audio_file
            start_idx = random.randint(0, max_start)
        
        # 获取连续帧
        continuous_frames = []
        for i in range(self.num_frames):
            if start_idx + i >= len(frame_paths):
                self.blacklist_sequences.add(sequence_name)
                return None, None, None, audio_file
            img = self.read_image(frame_paths[start_idx + i])
            if img is None:
                self.blacklist_sequences.add(sequence_name)
                return None, None, None, audio_file
            continuous_frames.append(img)
        
        # 获取参考帧（从不同位置选择）
        max_attempts = 5  # 减少尝试次数以提高速度
        for _ in range(max_attempts):
            if len(frame_paths) <= self.num_frames:
                self.blacklist_sequences.add(sequence_name)
                return None, None, None, audio_file
                
            wrong_start_idx = random.randint(0, len(frame_paths) - self.num_frames)
            # 确保参考帧与连续帧不重叠
            if wrong_start_idx > start_idx - self.num_frames and wrong_start_idx < start_idx + self.num_frames:
                continue
                
            ref_frames = []
            valid_ref = True
            for i in range(self.num_frames):
                if wrong_start_idx + i >= len(frame_paths):
                    valid_ref = False
                    break
                img = self.read_image(frame_paths[wrong_start_idx + i])
                if img is None:
                    valid_ref = False
                    break
                ref_frames.append(img)
            
            if valid_ref:
                return np.array(continuous_frames), np.array(ref_frames), start_idx, audio_file
        
        # 如果找不到有效的参考帧，将序列加入黑名单
        self.blacklist_sequences.add(sequence_name)
        return None, None, None, audio_file
    
    def worker_init_fn(self, worker_id):
        """初始化工作进程"""
        self.worker_id = worker_id
        # 设置PyTorch的线程数，避免过度并行化
        torch.set_num_threads(1)
        
        # NumPy线程控制 - 兼容不同版本的NumPy
        try:
            import numpy as np
            # 尝试不同的NumPy线程控制方法
            if hasattr(np, 'set_num_threads'):
                np.set_num_threads(1)
            elif hasattr(np, 'core') and hasattr(np.core, 'set_num_threads'):
                np.core.set_num_threads(1)  # 有些版本在np.core中
            # 环境变量方式控制线程数
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
        except Exception as e:
            print(f"设置NumPy线程数时出错: {e}")
        
        setattr(
            self,
            f"image_processor_{worker_id}",
            ImageProcessor(self.resolution, self.mask, mask_image=self.mask_image),
        )
    
    def __getitem__(self, idx):
        """获取数据样本"""
        image_processor = getattr(self, f"image_processor_{self.worker_id}")
        max_attempts = 5  # 减少尝试次数以提高速度
        
        for attempt in range(max_attempts):
            try:
                # 选择一个有效的序列ID
                valid_sequences = [i for i, s in enumerate(self.sequences) 
                                  if s not in self.blacklist_sequences]
                if not valid_sequences:
                    raise RuntimeError("没有有效的序列可用")
                    
                sequence_id = random.choice(valid_sequences)
                
                # 获取连续帧和参考帧
                continuous_frames, ref_frames, start_idx, audio_file = self.get_sequence(sequence_id)
                if continuous_frames is None or ref_frames is None:
                    continue
                
                # 处理音频数据
                if self.load_audio_data:
                    sequence_name = self.sequences[sequence_id]
                    mel_cache_path = os.path.join(
                        self.audio_mel_cache_dir, f"{sequence_name}_mel.pt"
                    )
                    
                    try:
                        if os.path.isfile(mel_cache_path):
                            original_mel = torch.load(mel_cache_path)
                        else:
                            original_mel = self.read_audio_file(audio_file)
                            if original_mel is not None:
                                torch.save(original_mel, mel_cache_path)
                            else:
                                continue
                        
                        mel = self.crop_audio_window(original_mel, start_idx)
                        if mel is None or mel.shape[-1] != self.mel_window_length:
                            continue
                    except Exception:
                        continue
                else:
                    mel = []
                
                # 使用图像处理器准备遮罩和遮罩图像
                gt, masked_gt, mask = image_processor.prepare_masks_and_masked_images(
                    continuous_frames, affine_transform=False
                )
                
                if self.mask == "fix_mask":
                    ref, _, _ = image_processor.prepare_masks_and_masked_images(ref_frames, affine_transform=False)
                else:
                    ref = image_processor.process_images(ref_frames)
                
                break
            
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"在 {max_attempts} 次尝试后无法加载有效样本: {str(e)}")
                continue
        
        # 返回样本字典
        sample = dict(
            gt=gt,
            masked_gt=masked_gt,
            ref=ref,
            mel=mel,
            mask=mask,
            video_path=audio_file,
            start_idx=start_idx,
        )
        
        return sample