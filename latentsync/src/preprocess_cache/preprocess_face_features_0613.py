import argparse
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline_teacache_0603 import LipsyncPipeline
from diffusers.utils.import_utils import is_xformers_available
from latentsync.whisper.audio2feature import Audio2Feature
import os
import glob
import pickle
import gzip
import json

def build_pipeline(config, args):
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")
    if hasattr(args, 'whisper_model_path') and args.whisper_model_path:
        whisper_model_path = args.whisper_model_path

    audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,
        device="cpu",
    )
    unet = unet.to(dtype=dtype)

    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")
    return pipeline

def preprocess_and_save_features(config, args):
    os.makedirs(args.output_dir, exist_ok=True)
    pipeline = build_pipeline(config, args)
    pipeline.image_processor.device = "cuda"

    video_files = glob.glob(os.path.join(args.input_dir, "*.mp4"))
    print(f"共找到{len(video_files)}个视频文件。")
    print(f"video_files: {video_files}")
    # exit(0)
    if args.max_videos != -1:
        video_files = video_files[:args.max_videos]
        print(f"只处理前{args.max_videos}个视频。")

    index_list = []
    for video_path in video_files:
        base = os.path.basename(video_path)
        if not base.endswith(".mp4"):
            continue
        out_name = base.replace(".mp4", "_features.pkl")
        out_path = os.path.join(args.output_dir, out_name)
        if os.path.exists(out_path):
            print(f"已存在，跳过: {out_path}")
            try:
                with gzip.open(out_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    num_frames = cache_data.get('num_frames', None)
            except Exception:
                num_frames = None
            index_list.append({
                'video_path': video_path,
                'feature_path': out_path,
                'num_frames': num_frames
            })
            continue
        print(f"处理: {video_path}")
        faces, original_video_frames, boxes, affine_matrices = pipeline.affine_transform_video(video_path)
        cache_data = {
            'faces': faces.cpu().half() if torch.is_tensor(faces) else faces,
            'boxes': boxes,
            'affine_matrices': affine_matrices,
            'video_path': video_path,
            'num_frames': len(faces)
        }
        with gzip.open(out_path, 'wb', compresslevel=6) as f:
            pickle.dump(cache_data, f)
        print(f"已保存: {out_path}")
        index_list.append({
            'video_path': video_path,
            'feature_path': out_path,
            'num_frames': len(faces)
        })
    # 输出json索引
    json_dir = os.path.dirname(args.output_dir) if os.path.isdir(args.output_dir) else "."
    json_path = os.path.join(json_dir, "features_index.json")
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(index_list, jf, ensure_ascii=False, indent=2)
    print(f"特征索引已保存到: {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_videos", type=int, default=-1)
    parser.add_argument("--whisper_model_path", type=str, default=None)
    args = parser.parse_args()
    config = OmegaConf.load(args.unet_config_path)
    preprocess_and_save_features(config, args)

    # cd /home/work/hy_01/mmuplt-avatarv3/latent_sync
    # export PYTHONPATH=/home/work/hy_01/mmuplt-avatarv3/latent_sync:$PYTHONPATH
    # python /home/work/hy_01/mmuplt-avatarv3/latent_sync/latentsync/pipelines/preprocess_face_features_0613.py --max_videos 2