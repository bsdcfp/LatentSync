#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视频序列合并器
视频序列处理和合并工具，支持基于音频时长的视频帧重采样和拼接

Author: wang.li@shopee.com
Created: 2025-06-11
Description: 用于将多个视频序列合并成一个视频，并根据音频时长调整帧数
"""

import cv2
import numpy as np
import os
import wave
import librosa
from typing import List, Tuple
import argparse
import json
import random
import time


class VideoSelectedInfo:
    def __init__(self, connected_json_path, video_info_json_path, shot_video_json_path, debug=False):
        self.connected_info = self.read_json_file(connected_json_path)
        self.video_info = self.read_json_file(video_info_json_path)
        self.shot_video_info = self.read_json_file(shot_video_json_path)
        self.fps = 25
        self.debug = debug

        self.shot_video_duration_list = [int(key) for key in self.shot_video_info.keys()]
        
        # 创建独立的随机生成器，使用当前时间戳作为种子
        # 这样可以确保视频选择永远随机，不受外部随机种子影响
        self.video_random = random.Random()
        self.video_random.seed(int(time.time() * 1000000) % (2**32))  # 使用微秒时间戳
        
        if self.debug:
            print(f"🎲 视频选择器使用独立随机种子: {int(time.time() * 1000000) % (2**32)}")
        
        self.start_videos_list = {
            "connect_start": [
                "s01_01__se_se01_01_v2",
                "s01_02__se_se01_01_v1",
                "s01_03__se_se01_01_v1",
                "s01_04__se_se01_01_v1",
                "s01_05__se_se01_01_v1",
                "s01_06__se_se01_01_v1",
            ],
            "connect_next": [
                "s01_01__se_se01_02_v6",
                "s01_02__se_se01_02_v1",
                "s01_03__se_se01_02_v1",
                "s01_04__se_se01_02_v1",
                "s01_05__se_se01_02_v1",
                "s01_06__se_se01_02_v1",
            ],
        }

    def read_json_file(self, json_path):
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data

    def search_video_path(self, target_video_length):
        if target_video_length == 0:
            return []
        elif target_video_length == 1:
            if self.debug:
                print("target_video_length == 1")
            result_list = [self.video_random.choice(self.start_videos_list["connect_start"])]
            if self.debug:
                print(f"result_list:{result_list}")
            return result_list
        else:
            current_index = 1
            result_list = [self.video_random.choice(self.start_videos_list["connect_next"])]

            current_index += 1
            while current_index <= target_video_length:
                if current_index != target_video_length:
                    if current_index % 3 == 0:
                        current_video = self.video_random.choice(
                            self.connected_info[result_list[-1]]["connect_start"]
                        )
                    else:
                        current_video = self.video_random.choice(
                            self.connected_info[result_list[-1]]["connect_next"]
                        )
                    result_list.append(current_video)
                else:
                    result_list.append(
                        self.video_random.choice(
                            self.connected_info[result_list[-1]]["connect_start"]
                        )
                    )
                current_index += 1
            return result_list

    def get_video_name_list(self, audio_duration=23.5, add_short_audio_duration_th=19):
        target_frame_count = int(audio_duration * self.fps)
        lower_length, upper_length = (
            int(audio_duration // 7),
            int(audio_duration // 7) + 1,
        )
        # print(lower_length, upper_length)
        lower_length_result = self.search_video_path(lower_length)
        upper_length_result = self.search_video_path(upper_length)
        if self.debug:
            print(f"lower_length:{lower_length}, upper_length:{upper_length}")
            print(f"lower_length_result:{lower_length_result}, upper_length_result:{upper_length_result}")
        lower_length_result_list_info = {"frame_count": 0, "duration": 0}
        upper_length_result_list_info = {"frame_count": 0, "duration": 0}
        for video_name in lower_length_result:
            video_info = self.video_info[video_name]
            lower_length_result_list_info["frame_count"] += video_info["frame_count"]
            lower_length_result_list_info["duration"] += video_info["duration"]
        for video_name in upper_length_result:
            video_info = self.video_info[video_name]
            upper_length_result_list_info["frame_count"] += video_info["frame_count"]
            upper_length_result_list_info["duration"] += video_info["duration"]
        
        if audio_duration <= add_short_audio_duration_th:
            diff_duration = audio_duration - lower_length_result_list_info["duration"]
            if diff_duration / audio_duration >= 0.125:
                add_short_video_key = self.get_add_short_video_key(diff_duration, audio_duration, lower_length_result_list_info["duration"])
                add_short_video_name = random.choice(self.shot_video_info[add_short_video_key])
                lower_length_result.append(add_short_video_name)
                lower_length_result_list_info["frame_count"] += self.video_info[add_short_video_name]["frame_count"]
                lower_length_result_list_info["duration"] += self.video_info[add_short_video_name]["duration"]

        if self.debug:
            print(f"lower_length_result_list_info: {lower_length_result_list_info}, upper_length_result_list_info: {upper_length_result_list_info}")
        else:
            pass
        # 计算速率差
        upper_duration_gap = abs(
            upper_length_result_list_info["duration"] / audio_duration - 1
        )
        lower_duration_gap = abs(
            lower_length_result_list_info["duration"] / audio_duration - 1
        )
        if (
            lower_duration_gap < upper_duration_gap
            and lower_length_result_list_info["frame_count"] >= target_frame_count
        ):
            return lower_length_result
        else:
            return upper_length_result
        
    def get_add_short_video_key(self, diff_duration, audio_duration, selected_video_duration):
        assert 1 < diff_duration < 8
        lower_diff = min(self.shot_video_duration_list, key=lambda x: abs(x - diff_duration))
        upper_diff = min(self.shot_video_duration_list, key=lambda x: abs(x - diff_duration - 1))
        # print(f"lower_diff:{lower_diff}, upper_diff:{upper_diff}")
        if abs((selected_video_duration + lower_diff) / audio_duration -1) <= abs((selected_video_duration + upper_diff) / audio_duration -1):
            return str(lower_diff)
        else:
            return str(upper_diff)


class VideoSequenceMerger:
    """视频序列合并器"""

    def __init__(self, target_fps: int = 25, transition_frames: int = 10, debug=False):
        self.target_fps = target_fps
        self.transition_frames = transition_frames
        self.debug = debug

    def get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长（秒）"""
        # 尝试使用librosa获取音频时长
        duration = librosa.get_duration(path=audio_path)
        return duration

    def get_video_info(self, video_path: str) -> Tuple[int, float, int, int]:
        """获取视频信息：总帧数、时长、宽度、高度"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件 {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        cap.release()
        return frame_count, duration, width, height

    def merge_video_sequences(self, video_paths: List[str]) -> List[np.ndarray]:
        """合并视频序列，在接合点进行加权处理"""
        if not video_paths:
            raise ValueError("至少需要一个视频文件")

        all_frames = []
        previous_last_frame = None

        for i, video_path in enumerate(video_paths):
            print(f"处理视频 {i+1}/{len(video_paths)}: {video_path}")
            frames = self.read_video_frames(video_path)

            if not frames:
                raise ValueError(f"视频文件 {video_path} 没有有效帧")

            # 如果不是第一个视频，需要对第一帧进行加权处理
            # if previous_last_frame is not None:
            #     # 使用0.5权重混合上一个视频的最后一帧和当前视频的第一帧
            #     blended_first_frame = self.weighted_frame_blend(
            #         previous_last_frame, frames[0], weight=0.5
            #     )
            #     frames[0] = blended_first_frame

            all_frames.extend(frames)
            # previous_last_frame = frames[-1].copy()

        return all_frames

    def save_video(self, frames: List[np.ndarray], output_path: str, fps: int = 25):
        """保存帧序列为视频文件"""
        if not frames:
            raise ValueError("没有帧可以保存")

        height, width, channels = frames[0].shape
        
        if self.debug:
            print(f"💾 开始保存视频...")
            print(f"   - 帧数: {len(frames)}")
            print(f"   - 尺寸: {width}x{height}")
            print(f"   - 帧率: {fps} fps")
            print(f"   - 通道数: {channels}")
        
        # 使用更兼容的编码器
        # 尝试使用 H.264 编码器
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4
            cv2.VideoWriter_fourcc(*'XVID'),  # XVID
            cv2.VideoWriter_fourcc(*'H264'),  # H.264
            cv2.VideoWriter_fourcc(*'X264'),  # X264
        ]
        
        fourcc_names = ['mp4v', 'XVID', 'H264', 'X264']
        
        out = None
        used_codec = None
        for i, fourcc in enumerate(fourcc_options):
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                used_codec = fourcc_names[i]
                if self.debug:
                    print(f"   ✅ 使用编码器: {used_codec}")
                break
            out.release()
        
        if out is None or not out.isOpened():
            raise RuntimeError(f"无法创建视频写入器，输出路径: {output_path}")

        if self.debug:
            print(f"   🔄 正在写入 {len(frames)} 帧...")
            
        for i, frame in enumerate(frames):
            # 确保帧尺寸一致
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            out.write(frame)
            
            # 显示进度（每100帧显示一次）
            if self.debug and (i + 1) % 100 == 0:
                print(f"   📊 已写入 {i + 1}/{len(frames)} 帧 ({(i + 1)/len(frames)*100:.1f}%)")

        out.release()
        
        if self.debug:
            print(f"   ✅ 视频保存完成: {output_path}")
        else:
            print(f"视频已保存到: {output_path}")

    def process_video_sequences(
        self,
        video_paths: List[str],
        audio_path: str = None,
        audio_duration: float = None,
        output_path: str = None,
    ) -> None:
        """处理视频序列的主函数"""
        print("开始处理视频序列...")

        if audio_path is None and audio_duration is None:
            raise ValueError("音频路径和音频时长不能同时为空")

        # 获取音频时长
        if audio_duration is None:
            audio_duration = self.get_audio_duration(audio_path)
        print(f"音频时长: {audio_duration:.2f} 秒")

        # 计算需要的总帧数
        target_frame_count = int(audio_duration * self.target_fps)
        print(f"目标帧数: {target_frame_count} (基于 {self.target_fps} fps)")

        # 获取所有视频的信息
        total_frames = 0
        video_info = []

        for video_path in video_paths:
            frame_count, duration, width, height = self.get_video_info(video_path)
            video_info.append((frame_count, duration, width, height))
            total_frames += frame_count
            print(
                f"视频 {video_path}: {frame_count} 帧, {duration:.2f} 秒, {width}x{height}"
            )

        print(f"视频序列总帧数: {total_frames}")

        # 检查帧数是否足够
        if total_frames < target_frame_count:
            raise ValueError(
                f"视频序列总帧数 ({total_frames}) 小于音频所需帧数 ({target_frame_count})"
            )

        # 合并视频序列
        merged_frames = self.merge_video_sequences(video_paths)
        print(f"合并后总帧数: {len(merged_frames)}")

        # 重采样到目标帧数
        resampled_frames, original_index_list = self.resample_frames(
            merged_frames, target_frame_count
        )
        print(f"重采样后帧数: {len(resampled_frames)}")
        # print(f"重采样后帧序列: {len(original_index_list)}")

        # 保存输出视频（如果指定了输出路径）
        if output_path is not None:
            self.save_video(resampled_frames, output_path, self.target_fps)
        else:
            print("未指定输出路径，跳过保存视频, 返回索引列表")

        # print("处理完成!")
        return original_index_list

    def resample_frame_indices(self, total_frame_count: int, target_frame_count: int) -> List[int]:
        """轻量化版本：只重采样帧索引，不处理实际帧数据
        
        参数:
            total_frame_count: 原始总帧数
            target_frame_count: 目标帧数
            
        返回:
            重采样后的帧索引列表
        """
        if total_frame_count == target_frame_count:
            return list(range(total_frame_count))

        if total_frame_count < target_frame_count:
            raise ValueError(
                f"原始帧数 ({total_frame_count}) 小于目标帧数 ({target_frame_count})"
            )
        
        original_index_list = [0]
        
        if target_frame_count <= 2:
            if target_frame_count == 2:
                original_index_list.append(total_frame_count - 1)
            return original_index_list[:target_frame_count]

        transition_frames = self.transition_frames

        if target_frame_count <= 2 * transition_frames + 1:
            # 如果总帧数太少，只能进行线性采样
            for i in range(1, target_frame_count - 1):
                ratio = i / (target_frame_count - 1)
                original_index = int(ratio * (total_frame_count - 1))
                original_index_list.append(original_index)
        else:
            # 第一段：采样间隔从1开始递增
            current_pos = 0.0
            max_interval = min(
                (total_frame_count - 1) / (target_frame_count - 1) * 2,
                total_frame_count / (transition_frames * 2),
            )

            for i in range(1, transition_frames + 1):
                # 间隔从1递增到max_interval
                interval = 1 + (max_interval - 1) * (i / transition_frames)
                current_pos += interval
                original_index = min(int(current_pos), total_frame_count - 1)
                original_index_list.append(original_index)
            
            # 中间部分：均匀采样间隔
            middle_target_count = target_frame_count - 2 * transition_frames - 2
            if middle_target_count > 0:
                # 计算中间段的起始和结束位置
                middle_start_pos = current_pos
                # 为最后一段预留空间
                remaining_frames = total_frame_count - 1 - current_pos
                last_section_frames = (
                    transition_frames * max_interval / 2
                )  # 估算最后一段需要的帧数
                middle_end_pos = total_frame_count - 1 - last_section_frames

                if middle_end_pos > middle_start_pos:
                    middle_interval = (
                        middle_end_pos - middle_start_pos
                    ) / middle_target_count
                    for i in range(middle_target_count):
                        current_pos = middle_start_pos + (i + 1) * middle_interval
                        original_index = min(int(current_pos), total_frame_count - 1)
                        original_index_list.append(original_index)
            
            # 最后一段：采样间隔从max_interval递减到1
            # 重新计算当前位置，确保能到达最后一帧
            frames_to_end = total_frame_count - 1 - current_pos
            total_interval_sum = sum(
                max_interval - (max_interval - 1) * (i / transition_frames)
                for i in range(transition_frames)
            )

            if total_interval_sum > 0:
                scale_factor = frames_to_end / total_interval_sum
                for i in range(transition_frames):
                    # 间隔从max_interval递减到1
                    interval = max_interval - (max_interval - 1) * (
                        i / transition_frames
                    )
                    current_pos += interval * scale_factor
                    original_index = min(int(current_pos), total_frame_count - 1)
                    original_index_list.append(original_index)
        
        # 最后一帧
        original_index_list.append(total_frame_count - 1)
        return original_index_list[:target_frame_count]

    def resample_frame_indices_segmented(
        self, video_frame_counts: List[int], target_frame_count: int
    ) -> List[int]:
        """分段重采样：为每个视频分配帧数，然后单独重采样
        
        参数:
            video_frame_counts: 每个视频的帧数列表
            target_frame_count: 目标总帧数
            
        返回:
            重采样后的帧索引列表
        """
        if self.debug:
            print(f"🔄 开始分段重采样，视频数量: {len(video_frame_counts)}")
            
        total_original_frames = sum(video_frame_counts)
        
        # 步骤1：按比例分配每个视频应该贡献的帧数
        allocated_frames = []
        allocated_total = 0
        
        for i, original_frames in enumerate(video_frame_counts):
            # 按比例计算应分配的帧数
            proportion = original_frames / total_original_frames
            allocated = int(target_frame_count * proportion)
            
            # 确保至少分配1帧（如果原视频有帧的话）
            if original_frames > 0 and allocated == 0:
                allocated = 1
                
            allocated_frames.append(allocated)
            allocated_total += allocated
            
            if self.debug:
                print(f"   视频 {i}: 原始 {original_frames} 帧 -> 分配 {allocated} 帧 (比例: {proportion:.3f})")
        
        # 调整分配，确保总数等于目标帧数
        difference = target_frame_count - allocated_total
        if difference != 0:
            if self.debug:
                print(f"   调整帧数差异: {difference}")
            
            # 优先调整帧数最多的视频
            if difference > 0:
                # 需要增加帧数，给帧数最多的视频增加
                for _ in range(difference):
                    max_idx = allocated_frames.index(max(allocated_frames))
                    allocated_frames[max_idx] += 1
            else:
                # 需要减少帧数，从帧数最多的视频减少
                for _ in range(-difference):
                    max_idx = allocated_frames.index(max(allocated_frames))
                    if allocated_frames[max_idx] > 1:  # 确保不会减到0以下
                        allocated_frames[max_idx] -= 1
        
        if self.debug:
            print(f"   最终分配: {allocated_frames}, 总计: {sum(allocated_frames)}")
        
        # 步骤2：对每个视频段单独进行重采样
        all_indices = []
        current_offset = 0
        
        for i, (original_frames, target_frames) in enumerate(zip(video_frame_counts, allocated_frames)):
            if target_frames == 0:
                continue
                
            if original_frames <= target_frames:
                # 如果目标帧数大于等于原始帧数，取所有帧
                segment_indices = list(range(original_frames))
                # 如果还需要更多帧，重复最后一帧
                while len(segment_indices) < target_frames:
                    segment_indices.append(original_frames - 1)
            else:
                # 如果需要减少帧数，使用简单的均匀采样
                segment_indices = self._uniform_sample_indices(original_frames, target_frames)
            
            # 将相对索引转换为绝对索引
            absolute_indices = [idx + current_offset for idx in segment_indices]
            all_indices.extend(absolute_indices)
            
            if self.debug:
                print(f"   视频 {i}: 重采样 {original_frames} -> {len(segment_indices)} 帧, 偏移 {current_offset}")
            
            current_offset += original_frames
        
        if self.debug:
            print(f"✅ 分段重采样完成，总帧数: {len(all_indices)}")
            
        return all_indices
    
    def _uniform_sample_indices(self, total_frames: int, target_frames: int) -> List[int]:
        """均匀采样帧索引
        
        参数:
            total_frames: 总帧数
            target_frames: 目标帧数
            
        返回:
            采样后的帧索引列表
        """
        if target_frames >= total_frames:
            return list(range(total_frames))
        
        if target_frames == 1:
            return [0]
        
        # 均匀分布采样
        indices = []
        step = (total_frames - 1) / (target_frames - 1)
        
        for i in range(target_frames):
            idx = int(round(i * step))
            idx = min(idx, total_frames - 1)  # 确保不超出范围
            indices.append(idx)
        
        return indices

    def process_video_sequences_lightweight(
        self,
        video_names_list: List[str],
        video_info_dict: dict,
        audio_path: str = None,
        audio_duration: float = None,
    ) -> List[int]:
        """轻量化版本：只处理帧数和索引，不读取/保存视频"""
        if self.debug:
            print("开始轻量化处理视频序列...")

        if audio_path is None and audio_duration is None:
            raise ValueError("音频路径和音频时长不能同时为空")

        # 获取音频时长
        if audio_duration is None:
            audio_duration = self.get_audio_duration(audio_path)
        if self.debug:
            print(f"音频时长: {audio_duration:.2f} 秒")

        # 计算需要的总帧数
        target_frame_count = int(audio_duration * self.target_fps)
        if self.debug:
            print(f"目标帧数: {target_frame_count} (基于 {self.target_fps} fps)")

        # 获取所有视频的帧数信息
        total_frames = 0
        video_frame_counts = []
        for video_name in video_names_list:
            frame_count = video_info_dict[video_name]["frame_count"]
            duration = video_info_dict[video_name]["duration"]
            width = video_info_dict[video_name]["width"]
            height = video_info_dict[video_name]["height"]
            total_frames += frame_count
            video_frame_counts.append(frame_count)
            if self.debug:
                print(f"视频 {video_name}: {frame_count} 帧, {duration:.2f} 秒, {width}x{height}")

        if self.debug:
            print(f"视频序列总帧数: {total_frames}")

        # 检查帧数是否足够
        if total_frames < target_frame_count:
            raise ValueError(
                f"视频序列总帧数 ({total_frames}) 小于音频所需帧数 ({target_frame_count})"
            )

        if self.debug:
            print(f"合并后总帧数: {total_frames}")

        # 使用分段重采样策略
        original_index_list = self.resample_frame_indices_segmented(
            video_frame_counts, target_frame_count
        )
        if self.debug:
            print(f"重采样后帧数: {len(original_index_list)}")
            print(f"original_index_list: {original_index_list}")

        return original_index_list


class VideoIndexGenerator:
    """视频索引生成器 - 三合一封装类"""
    
    def __init__(self, connected_json_path, video_info_json_path, shot_video_json_path, video_template_dir, 
                 target_fps=25, transition_frames=20, debug=False):
        self.video_template_dir = video_template_dir
        self.target_fps = target_fps
        self.transition_frames = transition_frames
        self.debug = debug
        
        # 初始化视频选择器
        self.video_selector = VideoSelectedInfo(connected_json_path, video_info_json_path, shot_video_json_path,debug)
        
        # 初始化视频合并器
        self.video_merger = VideoSequenceMerger(target_fps, transition_frames, debug)
    
    # @profile
    def get_video_index(self, audio_path, audio_duration=None, output_path=None):
        """
        根据音频路径生成视频索引
        
        参数:
            audio_path: 音频文件路径
            audio_duration: 音频时长（可选，如果不提供会自动计算）
            output_path: 输出视频路径（可选，如果提供则会合并视频并输出）
            
        返回:
            list of dict，每个dict包含video_name、frame_indices和video_path
        """
        video_info_dict = self.video_selector.video_info

        if not os.path.exists(audio_path):
            print(f"错误: 音频文件不存在: {audio_path}")
            return None
            
        if audio_duration is None:
            audio_duration = self.video_merger.get_audio_duration(audio_path)
            
        # if self.debug:
            print(f"音频时长: {audio_duration} 秒")

        video_names_list = self.video_selector.get_video_name_list(audio_duration)
        
        if self.debug:
            print(f"视频序列长度: {len(video_names_list)}, 视频序列: {video_names_list}")

        # 使用轻量化版本，只处理帧数和索引
        merged_index_list = self.video_merger.process_video_sequences_lightweight(
            video_names_list, video_info_dict, audio_path, audio_duration
        )
        if self.debug:
            print(f"merged_index_list: {merged_index_list}")
        # 将索引映射回原始视频
        original_video_index = self._map_indices_to_original_videos(
            merged_index_list, video_names_list, video_info_dict
        )

        # 如果提供了输出路径，则合并视频并输出
        if output_path is not None:
            self._merge_and_save_video(original_video_index, output_path)
        if self.debug:
            print(f"video_index_list: {original_video_index}")
            print(f"📋 获取的视频索引列表:")
            total_frames = 0
            for item in original_video_index:
                frame_count = len(item['frame_indices'])
                total_frames += frame_count
                print(f"  - {item['video_name']}: {frame_count} 帧")
            print(f"  📊 总计: {total_frames} 帧")
        return original_video_index
    
    def _map_indices_to_original_videos(self, merged_indices, video_names_list, video_info_dict):
        """
        将合并后的视频帧索引映射回原始视频序列

        参数:
            merged_indices: 合并后视频的帧索引列表
            video_names_list: 视频名称序列列表
            video_info_dict: 包含视频信息的字典

        返回:
            list of dict，每个dict包含video_name、frame_indices和video_path
        """
        # 使用list保持严格顺序，避免重复视频名覆盖问题
        video_ranges = []
        current_offset = 0
        # print(f"video_names_list: {video_names_list}")

        for i, video_name in enumerate(video_names_list):
            frame_count = video_info_dict[video_name]["frame_count"]
            video_ranges.append({
                'video_name': video_name,
                'start': current_offset,
                'end': current_offset + frame_count,
                'frame_count': frame_count
            })
            current_offset += frame_count
            if self.debug:
                print(f"   位置 {i} - 视频 {video_name}: start: {video_ranges[i]['start']} - end: {video_ranges[i]['end']}, 帧数 {frame_count}")
        
        # 为每个位置初始化结果列表
        position_results = []
        for i in range(len(video_names_list)):
            position_results.append([])

        # 遍历合并后的索引，将它们映射回原始视频
        for merged_index in merged_indices:
            # 找到这个索引属于哪个位置的视频
            target_position = None
            for position in range(len(video_names_list)):
                video_range = video_ranges[position]
                if video_range['start'] <= merged_index < video_range['end']:
                    target_position = position
                    break
            
            if target_position is None:
                # if self.debug:
                print(f"   ⚠️ 警告: 无法为索引 {merged_index} 找到对应的视频")
                continue
                
            # 计算在原始视频中的相对索引
            video_range = video_ranges[target_position]
            original_index = merged_index - video_range['start']
            
            # 检查索引是否在有效范围内
            max_frame_index = video_range['frame_count'] - 1
            if original_index > max_frame_index:
                # if self.debug:
                print(f"   ⚠️ 警告: 位置 {target_position} 视频 {video_range['video_name']} 的索引 {original_index} 超出范围 (最大: {max_frame_index})")
                # 将超出范围的索引截断到最大有效索引
                # original_index = max_frame_index
            
            position_results[target_position].append(original_index)

        # print(f"position_results: {position_results}")
        # 将结果改为数组形式，保持每个位置的视频段独立（不按视频名合并）
        final_result = []
        for position in range(len(position_results)):
            frame_indices = position_results[position]
            if len(frame_indices) == 0:
                continue
                
            video_name = video_ranges[position]['video_name']
            final_result.append({
                "video_name": video_name, 
                "frame_indices": sorted(frame_indices),  # 排序确保索引有序
                "video_path": os.path.join(self.video_template_dir, video_name + ".mp4")
            })

        # 直接返回数组结果
        list_of_dict = final_result
        
        if self.debug:
            print("   📋 索引映射结果:")
            for item in list_of_dict:
                frame_count = len(item['frame_indices'])
                if frame_count > 0:
                    min_idx = min(item['frame_indices'])
                    max_idx = max(item['frame_indices'])
                    max_allowed = video_info_dict[item['video_name']]["frame_count"] - 1
                    print(f"     - {item['video_name']}: {frame_count} 帧, 索引范围 [{min_idx}-{max_idx}], 最大允许: {max_allowed}")
        
        return list_of_dict

    def _merge_and_save_video(self, video_index_list, output_path):
        """
        根据视频索引列表合并视频并保存
        
        参数:
            video_index_list: 视频索引列表，每个元素包含video_name、frame_indices和video_path
            output_path: 输出视频路径
        """
        if self.debug:
            print(f"🎬 开始合并视频，输出到: {output_path}")
            
        all_frames = []
        total_frames = 0
        target_width = None
        target_height = None
        
        for video_info in video_index_list:
            video_path = video_info["video_path"]
            frame_indices = video_info["frame_indices"]
            video_name = video_info["video_name"]
            
            if self.debug:
                print(f"📹 处理视频: {video_name}")
                print(f"   需要帧数: {len(frame_indices)}")
                print(f"   帧索引范围: {min(frame_indices)} - {max(frame_indices)}")
            
            # 检查视频文件是否存在
            if not os.path.exists(video_path):
                print(f"⚠️  警告: 视频文件不存在: {video_path}")
                continue
                
            # 读取视频帧
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"⚠️  警告: 无法打开视频文件: {video_path}")
                continue
                
            video_frames = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 获取视频详细信息
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            video_duration = frame_count / video_fps if video_fps > 0 else 0
            
            if self.debug:
                print(f"   原视频信息:")
                print(f"     - 尺寸: {video_width}x{video_height}")
                print(f"     - 帧率: {video_fps:.2f} fps")
                print(f"     - 总帧数: {frame_count}")
                print(f"     - 时长: {video_duration:.2f} 秒")
            
            # 设置目标尺寸（使用第一个视频的尺寸）
            if target_width is None:
                target_width = video_width
                target_height = video_height
                if self.debug:
                    print(f"🎯 设置目标输出尺寸: {target_width}x{target_height}")
            
            # 读取所有帧
            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                # 统一帧尺寸
                if frame.shape[:2] != (target_height, target_width):
                    frame = cv2.resize(frame, (target_width, target_height))
                video_frames.append(frame)
            cap.release()
            
            if self.debug:
                print(f"   ✅ 成功读取 {len(video_frames)} 帧")
            
            # 根据索引提取需要的帧
            selected_frames = []
            for idx in frame_indices:
                if 0 <= idx < len(video_frames):
                    selected_frames.append(video_frames[idx])
                else:
                    print(f"⚠️  警告: 帧索引 {idx} 超出范围 (0-{len(video_frames)-1})")
            
            all_frames.extend(selected_frames)
            total_frames += len(selected_frames)
            
            if self.debug:
                print(f"   📊 提取了 {len(selected_frames)} 帧")
        
        if not all_frames:
            print("❌ 错误: 没有有效的帧可以合并")
            return
            
        # 计算输出视频信息
        output_duration = total_frames / self.target_fps
        
        if self.debug:
            print(f"\n📈 合并视频统计信息:")
            print(f"   - 总帧数: {total_frames}")
            print(f"   - 输出帧率: {self.target_fps} fps")
            print(f"   - 输出时长: {output_duration:.2f} 秒")
            print(f"   - 输出尺寸: {target_width}x{target_height}")
            print(f"   - 输出路径: {output_path}")
        
        # 保存合并后的视频
        self.video_merger.save_video(all_frames, output_path, self.target_fps)
        
        if self.debug:
            print(f"\n🎉 视频合并完成!")
            print(f"📄 最终输出视频信息:")
            print(f"   - 文件路径: {output_path}")
            print(f"   - 视频尺寸: {target_width}x{target_height}")
            print(f"   - 视频帧率: {self.target_fps} fps")
            print(f"   - 视频时长: {output_duration:.2f} 秒")
            print(f"   - 总帧数: {total_frames}")
            
            # 检查文件是否成功生成
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"   - 文件大小: {file_size / (1024*1024):.2f} MB")
            else:
                print(f"   ⚠️  警告: 输出文件未找到")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='视频序列合并器')
    parser.add_argument('--debug', action='store_true', help='开启调试模式，打印详细信息')
    parser.add_argument('--output_path', type=str, help='输出视频路径（可选，如果提供则会合并视频并输出）')
    args = parser.parse_args()
    
    # 配置参数
    audio_path = "/Users/wang.li/Downloads/test_data/audios/t_7.WAV"
    video_template_dir = "/Users/wang.li/Downloads/0612_online_templete_female_32fps_resize_720_1560/videos"
    connected_json_path = "/Users/wang.li/Downloads/connected_videos.json"
    video_info_json_path = "/Users/wang.li/Downloads/0612_video_templete_32fps_info_v2.json"
    shot_video_json_path = '/Users/wang.li/Downloads/connected_videos_other_short.json'

    audio_duration = 24.5

    # 使用新的三合一类
    video_index_generator = VideoIndexGenerator(
        connected_json_path, 
        video_info_json_path, 
        shot_video_json_path,
        video_template_dir,
        target_fps=25, 
        transition_frames=20, 
        debug=args.debug
    )

    result = video_index_generator.get_video_index(audio_path, audio_duration=audio_duration, output_path=args.output_path)
    print("视频索引结果:")
    print(result)
    
    if args.output_path:
        print(f"\n视频已合并并保存到: {args.output_path}")
    
    # python /home/work/hy_01/mmuplt-avatarv3/video_scripts_online/video_sequence_merger.py --debug --output_path /home/work/hy_01/mmuplt-avatarv3/video_scripts_online/output.mp4


    # 或者使用原有方式（兼容旧代码）
    # merger = VideoSequenceMerger(
    #     target_fps=25, transition_frames=20, debug=args.debug
    # )
    # video_selected = VideoSelectedInfo(
    #     connected_json_path, video_info_json_path, debug=args.debug
    # )
    # original_video_index_dict = get_video_index(
    #     audio_path, video_template_dir, merger, video_selected, debug=args.debug
    # )
    # print(original_video_index_dict)
