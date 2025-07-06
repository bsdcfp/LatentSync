import gc
import json
import os
import random
import sys
import time
from functools import partial
from glob import glob

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, logging as tf_logging

from IndexKits.index_kits import ResolutionGroup
from IndexKits.index_kits.sampler import DistributedSamplerWithStartIndex, BlockDistributedSampler
from hydit.config import get_args
from hydit.constants import VAE_EMA_PATH, TEXT_ENCODER, TOKENIZER, T5_ENCODER
from hydit.data_loader.arrow_load_stream import TextImageArrowStream
from hydit.diffusion import create_diffusion
from hydit.ds_config import deepspeed_config_from_args
from hydit.lr_scheduler import WarmupLR
from hydit.modules.ema import EMA
from hydit.modules.fp16_layers import Float16Module
from hydit.modules.models import HUNYUAN_DIT_MODELS, HunYuanDiT
from hydit.modules.text_encoder import MT5Embedder
from hydit.modules.posemb_layers import init_image_posemb
from hydit.utils.tools import create_exp_folder, model_resume, get_trainable_params

from diffusers import DDPMScheduler
from pathlib import Path


class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps

        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev


def load_torch_weights(model, dit_weight):

        load_key = 'module'
        if dit_weight is not None:
            dit_weight = Path(dit_weight)
            if dit_weight.is_dir():
                files = list(dit_weight.glob("*.pt"))
                if len(files) == 0:
                    raise ValueError(f"No model weights found in {dit_weight}")
                if str(files[0]).startswith('pytorch_model_'):
                    model_path = dit_weight / f"pytorch_model_{load_key}.pt"
                    bare_model = True
                elif any(str(f).endswith('_model_states.pt') for f in files):
                    files = [f for f in files if str(f).endswith('_model_states.pt')]
                    model_path = files[0]
                    bare_model = False
                else:
                    raise ValueError(f"Invalid model path: {dit_weight} with unrecognized weight format: "
                                     f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                                     f"`pytorch_model_*.pt`(provided by HunyuanDiT official) and "
                                     f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                                     f"specific weight file, please provide the full path to the file.")
            elif dit_weight.is_file():
                model_path = dit_weight
                bare_model = 'unknown'
            else:
                raise ValueError(f"Invalid model path: {dit_weight}")
        else:
            model_dir = self.root / "model"
            model_path = model_dir / f"pytorch_model_{load_key}.pt"
            bare_model = True

        if not model_path.exists():
            raise ValueError(f"model_path not exists: {model_path}")
        if model_path.suffix == '.safetensors':
            raise NotImplementedError(f"Loading safetensors is not supported yet.")
        else:
            # Assume it's a single weight file in the *.pt format.
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

        if bare_model == 'unknown' and ('ema' in state_dict or 'module' in state_dict):
            bare_model = False
        if bare_model is False:
            if load_key in state_dict:
                state_dict = state_dict[load_key]
            else:
                raise KeyError(f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                               f"are: {list(state_dict.keys())}.")

        if 'style_embedder.weight' in state_dict and not hasattr(model, 'style_embedder'):
            raise ValueError(f"You might be attempting to load the weights of HunYuanDiT version <= 1.1. You need "
                             f"to set `--use-style-cond --size-cond 1024 1024 --beta-end 0.03` to adapt to these weights."
                             f"Alternatively, you can use weights of version >= 1.2, which no longer depend on "
                             f"these two parameters.")
        if 'style_embedder.weight' not in state_dict and hasattr(model, 'style_embedder'):
            raise ValueError(f"You might be attempting to load the weights of HunYuanDiT version >= 1.2. You need "
                             f"to remove `--use-style-cond` and `--size-cond 1024 1024` to adapt to these weights.")
        # Don't set strict=False. Always explicitly check the state_dict.
        model.load_state_dict(state_dict, strict=True)
        return model


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def velocity_from_xstart_and_noise(x_start, timesteps, noise, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, x_start.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, x_start.shape)
    target = alphas * noise - sigmas * x_start
    return target



def get_predicted_original_sample(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_x_0

# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip, c_out

# Based on step 4 in DDIMScheduler.step
def get_predicted_noise(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_epsilon = (sample - alphas * model_output) / sigmas
    elif prediction_type == "v_prediction":
        pred_epsilon = alphas * model_output + sigmas * sample
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )
    return pred_epsilon

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

def deepspeed_initialize(args, logger, model, opt, deepspeed_config):
    logger.info(f"Initialize deepspeed...")
    logger.info(f"    Using deepspeed optimizer")

    def get_learning_rate_scheduler(warmup_min_lr, lr, warmup_num_steps, opt):
        return WarmupLR(opt, warmup_min_lr, lr, warmup_num_steps)

    logger.info(f"    Building scheduler with warmup_min_lr={args.warmup_min_lr}, warmup_num_steps={args.warmup_num_steps}")
    model, opt, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=get_trainable_params(model),
        config_params=deepspeed_config,
        args=args,
        lr_scheduler=partial(get_learning_rate_scheduler, args.warmup_min_lr, args.lr, args.warmup_num_steps) if args.warmup_num_steps > 0 else None,
    )
    return model, opt, scheduler

def save_checkpoint(args, rank, logger, model, ema, epoch, train_steps, checkpoint_dir, by='step'):
    def save_lora_weight(checkpoint_dir, client_state, tag=f"{train_steps:07d}.pt"):
        cur_ckpt_save_dir = f"{checkpoint_dir}/{tag}"
        if rank == 0:
            if args.use_fp16:
                model.module.module.save_pretrained(cur_ckpt_save_dir)
            else:
                model.module.save_pretrained(cur_ckpt_save_dir)

    def save_model_weight(client_state, tag):
        checkpoint_path = f"{checkpoint_dir}/{tag}"
        try:
            if args.training_parts == "lora":
                save_lora_weight(checkpoint_dir, client_state, tag=tag)
            else:
                model.save_checkpoint(checkpoint_dir, client_state=client_state, tag=tag)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Saved failed to {checkpoint_path}. {type(e)}: {e}")
            return False, ''
        return True, checkpoint_path

    client_state = {
        "steps": train_steps,
        "epoch": epoch,
        "args": args
    }
    if ema is not None:
        client_state['ema'] = ema.state_dict()

    # Save model weights by epoch or step
    dst_paths = []
    if by == 'epoch':
        tag = f"e{epoch:04d}.pt"
        dst_paths.append(save_model_weight(client_state, tag))
    elif by == 'step':
        if train_steps % args.ckpt_every == 0:
            tag = f"{train_steps:07d}.pt"
            dst_paths.append(save_model_weight(client_state, tag))
        if train_steps % args.ckpt_latest_every == 0 or train_steps == args.max_training_steps:
            tag = "latest.pt"
            dst_paths.append(save_model_weight(client_state, tag))
    elif by == 'final':
        tag = "final.pt"
        dst_paths.append(save_model_weight(client_state, tag))
    else:
        raise ValueError(f"Unknown save checkpoint method: {by}")

    saved = any([state for state, _ in dst_paths])
    if not saved:
        return False

    # Maybe clear optimizer states
    if not args.save_optimizer_state:
        dist.barrier()
        if rank == 0 and len(dst_paths) > 0:
            # Delete optimizer states to avoid occupying too much disk space.
            for dst_path in dst_paths:
                for opt_state_path in glob(f"{dst_path}/zero_*_optim_states.pt"):
                    os.remove(opt_state_path)

    return True


@torch.no_grad()
def prepare_model_inputs(args, batch, device, vae, text_encoder, text_encoder_t5, freqs_cis_img):
    image, text_embedding, text_embedding_mask, text_embedding_t5, text_embedding_mask_t5, kwargs = batch

    # clip & mT5 text embedding
    text_embedding = text_embedding.to(device)
    text_embedding_mask = text_embedding_mask.to(device)
    encoder_hidden_states = text_encoder(
        text_embedding.to(device),
        attention_mask=text_embedding_mask.to(device),
    )[0]
    text_embedding_t5 = text_embedding_t5.to(device).squeeze(1)
    text_embedding_mask_t5 = text_embedding_mask_t5.to(device).squeeze(1)
    with torch.no_grad():
        output_t5 = text_encoder_t5(
            input_ids=text_embedding_t5,
            attention_mask=text_embedding_mask_t5 if T5_ENCODER['attention_mask'] else None,
            output_hidden_states=True
        )
        encoder_hidden_states_t5 = output_t5['hidden_states'][T5_ENCODER['layer_index']].detach()

    # additional condition
    if args.size_cond:
        image_meta_size = kwargs['image_meta_size'].to(device)
    else:
        image_meta_size = None
    if args.use_style_cond:
        style = kwargs['style'].to(device)
    else:
        style = None

    if args.extra_fp16:
        image = image.half()

    # Map input images to latent space + normalize latents:
    image = image.to(device)
    vae_scaling_factor = vae.config.scaling_factor
    latents = vae.encode(image).latent_dist.sample().mul_(vae_scaling_factor)

    # positional embedding
    _, _, height, width = image.shape
    reso = f"{height}x{width}"
    cos_cis_img, sin_cis_img = freqs_cis_img[reso]

    # Model conditions
    model_kwargs = dict(
        encoder_hidden_states=encoder_hidden_states,
        text_embedding_mask=text_embedding_mask,
        encoder_hidden_states_t5=encoder_hidden_states_t5,
        text_embedding_mask_t5=text_embedding_mask_t5,
        image_meta_size=image_meta_size,
        style=style,
        cos_cis_img=cos_cis_img,
        sin_cis_img=sin_cis_img,
    )

    return latents, model_kwargs

def main(args):
    if args.training_parts == "lora":
        args.use_ema = False

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    dist.init_process_group("nccl")
    deepspeed.init_distributed()
    world_size = dist.get_world_size()
    batch_size = args.batch_size
    grad_accu_steps = args.grad_accu_steps
    print(world_size)
    print(batch_size)
    print(grad_accu_steps)
    global_batch_size = world_size * batch_size * grad_accu_steps

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    # seed = args.global_seed * world_size + rank
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    print(global_batch_size)
    deepspeed_config = deepspeed_config_from_args(args, global_batch_size)

    # Setup an experiment folder
    experiment_dir, checkpoint_dir, logger = create_exp_folder(args, rank)

    # Log all the arguments
    logger.info(sys.argv)
    logger.info(str(args))
    # Save to a json file
    args_dict = vars(args)
    args_dict['world_size'] = world_size
    # with open(f"{experiment_dir}/args.json", 'w') as f:
    #     json.dump(args_dict, f, indent=4)

    # Disable the message "Some weights of the model checkpoint at ... were not used when initializing BertModel."
    # If needed, just comment the following line.
    tf_logging.set_verbosity_error()

    # ===========================================================================
    # Building HYDIT
    # ===========================================================================

    logger.info("Building HYDIT Model.")

    # ---------------------------------------------------------------------------
    #   Training sample base size, such as 256/512/1024. Notice that this size is
    #   just a base size, not necessary the actual size of training samples. Actual
    #   size of the training samples are correlated with `resolutions` when enabling
    #   multi-resolution training.
    # ---------------------------------------------------------------------------
    image_size = args.image_size
    if len(image_size) == 1:
        image_size = [image_size[0], image_size[0]]
    if len(image_size) != 2:
        raise ValueError(f"Invalid image size: {args.image_size}")
    assert image_size[0] % 8 == 0 and image_size[1] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder). " \
                                                              f"got {image_size}"
    latent_size = [image_size[0] // 8, image_size[1] // 8]

    pretrained = '/home/work/donnhe2/HunyuanDiT/log_EXP/030-dit_g2_full_1024p/checkpoints/final.pt'
    # initialize model by deepspeed
    assert args.deepspeed, f"Must enable deepspeed in this script: train_deepspeed.py"
    with deepspeed.zero.Init(data_parallel_group=torch.distributed.group.WORLD,
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=deepspeed_config,
                             mpu=None,
                             enabled=args.zero_stage == 3):
        model = HUNYUAN_DIT_MODELS[args.model](args,
                                               input_size=latent_size,
                                               log_fn=logger.info,
                                               )

        model = load_torch_weights(model, pretrained)
        # model.load_state_dict(ckptpath, strict=True)
    # Multi-resolution / Single-resolution training.
    if args.multireso:
        resolutions = ResolutionGroup(image_size[0],
                                      align=16,
                                      step=args.reso_step,
                                      target_ratios=args.target_ratios).data
    else:
        resolutions = ResolutionGroup(image_size[0],
                                      align=16,
                                      target_ratios=['1:1']).data

    freqs_cis_img = init_image_posemb(args.rope_img,
                                      resolutions=resolutions,
                                      patch_size=model.patch_size,
                                      hidden_size=model.hidden_size,
                                      num_heads=model.num_heads,
                                      log_fn=logger.info,
                                      rope_real=args.rope_real,
                                      )

    # Create EMA model and convert to fp16 if needed.
    ema = None
    if args.use_ema:
        ema = EMA(args, model, device, logger)

    # Setup gradient checkpointing
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Setup FP16 main model:
    if args.use_fp16:
        model = Float16Module(model, args)
    logger.info(f"    Using main model with data type {'fp16' if args.use_fp16 else 'fp32'}")

    diffusion = create_diffusion(
        noise_schedule=args.noise_schedule,
        predict_type=args.predict_type,
        learn_sigma=args.learn_sigma,
        mse_loss_weight_type=args.mse_loss_weight_type,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        noise_offset=args.noise_offset,
    )

    teacher_model = HUNYUAN_DIT_MODELS[args.model](args,input_size=latent_size,log_fn=logger.info,)
    teacher_model = load_torch_weights(teacher_model, pretrained)

    

    # Setup VAE
    logger.info(f"    Loading vae from {VAE_EMA_PATH}")
    vae = AutoencoderKL.from_pretrained(VAE_EMA_PATH)
    # Setup BERT text encoder
    logger.info(f"    Loading Bert text encoder from {TEXT_ENCODER}")
    text_encoder = BertModel.from_pretrained(TEXT_ENCODER, False, revision=None)
    # Setup BERT tokenizer:
    logger.info(f"    Loading Bert tokenizer from {TOKENIZER}")
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER)
    # Setup T5 text encoder
    mt5_path = T5_ENCODER['MT5']
    embedder_t5 = MT5Embedder(mt5_path, torch_dtype=T5_ENCODER['torch_dtype'], max_length=args.text_len_t5)
    tokenizer_t5 = embedder_t5.tokenizer
    text_encoder_t5 = embedder_t5.model

    if args.extra_fp16:
        logger.info(f"    Using fp16 for extra modules: vae, text_encoder")
        vae = vae.half().to(device)
        text_encoder = text_encoder.half().to(device)
        text_encoder_t5 = text_encoder_t5.half().to(device)
        teacher_model = teacher_model.half().to(device)
    else:
        vae = vae.to(device)
        text_encoder = text_encoder.to(device)
        text_encoder_t5 = text_encoder_t5.to(device)
        teacher_model = teacher_model.to(device)

    teacher_model.requires_grad_(False)

    logger.info(f"    Optimizer parameters: lr={args.lr}, weight_decay={args.weight_decay}")
    logger.info("    Using deepspeed optimizer")
    opt = None

    # ===========================================================================
    # Building Dataset
    # ===========================================================================

    logger.info(f"Building Streaming Dataset.")
    logger.info(f"    Loading index file {args.index_file} (v2)")

    dataset = TextImageArrowStream(args=args,
                                   resolution=image_size[0],
                                   random_flip=args.random_flip,
                                   log_fn=logger.info,
                                   index_file=args.index_file,
                                   multireso=args.multireso,
                                   batch_size=batch_size,
                                   world_size=world_size,
                                   random_shrink_size_cond=args.random_shrink_size_cond,
                                   merge_src_cond=args.merge_src_cond,
                                   uncond_p=args.uncond_p,
                                   text_ctx_len=args.text_len,
                                   tokenizer=tokenizer,
                                   uncond_p_t5=args.uncond_p_t5,
                                   text_ctx_len_t5=args.text_len_t5,
                                   tokenizer_t5=tokenizer_t5,
                                   )
    if args.multireso:
        sampler = BlockDistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=args.global_seed,
                                          shuffle=False, drop_last=True, batch_size=batch_size)
    else:
        sampler = DistributedSamplerWithStartIndex(dataset, num_replicas=world_size, rank=rank, seed=args.global_seed,
                                                   shuffle=False, drop_last=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    logger.info(f"    Dataset contains {len(dataset):,} images.")
    logger.info(f"    Index file: {args.index_file}.")
    if args.multireso:
        logger.info(f'    Using MultiResolutionBucketIndexV2 with step {dataset.index_manager.step} '
                    f'and base size {dataset.index_manager.base_size}')
        logger.info(f'\n  {dataset.index_manager.resolutions}')

    # ===========================================================================
    # Loading parameter
    # ===========================================================================

    logger.info(f"Loading parameter")
    start_epoch = 0
    start_epoch_step = 0
    train_steps = 0
    # Resume checkpoint if needed
    if args.resume:
        model, ema, start_epoch, start_epoch_step, train_steps = model_resume(args, model, ema, logger, len(loader))

    if args.training_parts == "lora":
        loraconfig = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            target_modules=args.target_modules
        )
        if args.use_fp16:
            model.module = get_peft_model(model.module, loraconfig)
        else:
            model = get_peft_model(model, loraconfig)
        
    logger.info(f"    Training parts: {args.training_parts}")
    
    model, opt, scheduler = deepspeed_initialize(args, logger, model, opt, deepspeed_config)
    logger.info(" ****************************** Init finished ******************************")

    # ===========================================================================
    # Training
    # ===========================================================================

    model.train()
    if args.use_ema:
        ema.eval()

    print(f"    Worker {rank} ready.")
    dist.barrier()

    iters_per_epoch = len(loader)
    logger.info(" ****************************** Running training ******************************")
    logger.info(f"      Number GPUs:               {world_size}")
    logger.info(f"      Number training samples:   {len(dataset):,}")
    logger.info(f"      Number parameters:         {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"      Number trainable params:   {sum(p.numel() for p in get_trainable_params(model)):,}")
    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Iters per epoch:           {iters_per_epoch:,}")
    logger.info(f"      Batch size per device:     {batch_size}")
    logger.info(f"      Batch size all device:     {batch_size * world_size * grad_accu_steps:,} (world_size * batch_size * grad_accu_steps)")
    logger.info(f"      Gradient Accu steps:       {args.grad_accu_steps}")
    logger.info(f"      Total optimization steps:  {args.epochs * iters_per_epoch // grad_accu_steps:,}")

    logger.info(f"      Training epochs:           {start_epoch}/{args.epochs}")
    logger.info(f"      Training epoch steps:      {start_epoch_step:,}/{iters_per_epoch:,}")
    logger.info(f"      Training total steps:      {train_steps:,}/{min(args.max_training_steps, args.epochs * iters_per_epoch):,}")
    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Noise schedule:            {args.noise_schedule}")
    logger.info(f"      Beta limits:               ({args.beta_start}, {args.beta_end})")
    logger.info(f"      Learn sigma:               {args.learn_sigma}")
    logger.info(f"      Prediction type:           {args.predict_type}")
    logger.info(f"      Noise offset:              {args.noise_offset}")

    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Using EMA model:           {args.use_ema} ({args.ema_dtype})")
    if args.use_ema:
        logger.info(f"      Using EMA decay:           {ema.max_value if args.use_ema else None}")
        logger.info(f"      Using EMA warmup power:    {ema.power if args.use_ema else None}")
    logger.info(f"      Using main model fp16:     {args.use_fp16}")
    logger.info(f"      Using extra modules fp16:  {args.extra_fp16}")
    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Experiment directory:      {experiment_dir}")
    logger.info("    *******************************************************************************")

    if args.gc_interval > 0:
        gc.disable()
        gc.collect()

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()


    # 1. Create the noise scheduler and the desired noise schedule.
    noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", revision=None)

    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    alpha_schedule = alpha_schedule.to(device)
    sigma_schedule = sigma_schedule.to(device)
    

    # new solver
    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        timesteps=1000,
        ddim_timesteps=50,
    )

    solver = solver.to(device)

    # Training loop
    epoch = start_epoch
    while epoch < args.epochs:
        # Random shuffle dataset
        shuffle_seed = args.global_seed + epoch
        logger.info(f"    Start random shuffle with seed={shuffle_seed}")
        # Makesure all processors use the same seed to shuffle dataset.
        dataset.shuffle(seed=shuffle_seed, fast=True)
        logger.info(f"    End of random shuffle")

        # Move sampler to start_index
        if not args.multireso:
            start_index = start_epoch_step * world_size * batch_size
            if start_index != sampler.start_index:
                sampler.start_index = start_index
                # Reset start_epoch_step to zero, to ensure next epoch will start from the beginning.
                start_epoch_step = 0
                logger.info(f"      Iters left this epoch: {len(loader):,}")

        logger.info(f"    Beginning epoch {epoch}...")
        for batch in loader:
            latents, model_kwargs = prepare_model_inputs(args, batch, device, vae, text_encoder, text_encoder_t5, freqs_cis_img)

            # new added code
            # 2. Sample a random timestep for each image t_n from the ODE solver timesteps without bias.
            bsz = latents.shape[0]
            num_ddim_timesteps = 50
            topk = 1000 // num_ddim_timesteps
            index = torch.randint(0, num_ddim_timesteps, (bsz,), device=latents.device).long()
            start_timesteps = solver.ddim_timesteps[index]
            timesteps = start_timesteps - topk
            timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

            # 3. Get boundary scalings for start_timesteps and (end) timesteps.
            c_skip_start, c_out_start = scalings_for_boundary_conditions(
                start_timesteps, timestep_scaling=10.0
            )
            c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
            c_skip, c_out = scalings_for_boundary_conditions(
                timesteps, timestep_scaling=10.0
            )
            c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

            noise = torch.randn_like(latents)
            noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)
            noisy_model_input.to(device)

            noise_pred = model(noisy_model_input, start_timesteps, **model_kwargs)['x']

            target = velocity_from_xstart_and_noise(noisy_model_input, start_timesteps, noise, alpha_schedule, sigma_schedule)


            B, C = noisy_model_input.shape[:2]
            # assert_shape(noise_pred, (B, C * 2, *x_t.shape[2:]))
            noise_pred, model_var_values = torch.split(noise_pred, C, dim=1)

            loss1 = mean_flat((target - noise_pred) ** 2).mean()

            pred_x_0 = get_predicted_original_sample(
                    noise_pred,
                    start_timesteps,
                    noisy_model_input,
                    "v_prediction",
                    alpha_schedule,
                    sigma_schedule,
                )

            model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0


            # 8. Compute the conditional and unconditional teacher model predictions to get CFG estimates of the
            # predicted noise eps_0 and predicted original sample x_0, then run the ODE solver using these
            # estimates to predict the data point in the augmented PF-ODE trajectory corresponding to the next ODE
            # solver timestep.

            with torch.no_grad():
                cond_teacher_output = teacher_model(noisy_model_input, start_timesteps, **model_kwargs)['x']

                B, C = noisy_model_input.shape[:2]
                # assert_shape(noise_pred, (B, C * 2, *x_t.shape[2:]))
                cond_teacher_output, model_var_values = torch.split(cond_teacher_output, C, dim=1)

                cond_pred_x0 = get_predicted_original_sample(
                        cond_teacher_output,
                        start_timesteps,
                        noisy_model_input,
                        "v_prediction",
                        alpha_schedule,
                        sigma_schedule,
                    )
                
                cond_pred_noise = get_predicted_noise(
                        cond_teacher_output,
                        start_timesteps,
                        noisy_model_input,
                        "v_prediction",
                        alpha_schedule,
                        sigma_schedule,
                    )
                
                pred_x0 = cond_pred_x0
                pred_noise = cond_pred_noise
                x_prev = solver.ddim_step(pred_x0, pred_noise, index)


            x_prev = x_prev.to(noisy_model_input.dtype).to(device)
            # 9. Get target LCM prediction on x_prev, w, c, t_n (timesteps)
            with torch.no_grad():
                
                target_noise_pred = teacher_model(x_prev, timesteps, **model_kwargs)['x']

                B, C = x_prev.shape[:2]
                # assert_shape(noise_pred, (B, C * 2, *x_t.shape[2:]))
                target_noise_pred, model_var_values = torch.split(target_noise_pred, C, dim=1)

                pred_x_0 = get_predicted_original_sample(
                        target_noise_pred,
                        timesteps,
                        x_prev,
                        "v_prediction",
                        alpha_schedule,
                        sigma_schedule,
                    )

                target = c_skip * x_prev + c_out * pred_x_0

            
        
            # loss_dict = diffusion.training_losses(model=model, x_start=latents, model_kwargs=model_kwargs)
            # loss1 = loss_dict["loss"].mean()
            loss2 = torch.mean(torch.sqrt((model_pred.float() - target.float()) ** 2 + 0.001 ** 2) - 0.001)
            loss = loss1 * 0.0 + loss2 * 1
            # loss = loss2

            logger.info(f"      training loss:     {loss}")
            model.backward(loss)
            last_batch_iteration = (train_steps + 1) // (global_batch_size // (batch_size * world_size))
            model.step(lr_kwargs={'last_batch_iteration': last_batch_iteration})

            if args.use_ema:
                if args.use_fp16:
                    ema.update(model.module.module, step=train_steps)
                else:
                    ema.update(model.module, step=train_steps)

            # ===========================================================================
            # Log loss values:
            # ===========================================================================
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                # get lr from deepspeed fused optimizer
                logger.info(f"(step={train_steps:07d}) " +
                            (f"(update_step={train_steps // args.grad_accu_steps:07d}) " if args.grad_accu_steps > 1 else "") +
                            f"Train Loss: {avg_loss:.4f}, "
                            f"Lr: {opt.param_groups[0]['lr']:.6g}, "
                            f"Steps/Sec: {steps_per_sec:.2f}, "
                            f"Samples/Sec: {int(steps_per_sec * batch_size * world_size):d}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # collect gc:
            if args.gc_interval > 0 and (train_steps % args.gc_interval == 0):
                gc.collect()

            if (train_steps % args.ckpt_every == 0 or train_steps % args.ckpt_latest_every == 0
                ) and train_steps > 0:
                save_checkpoint(args, rank, logger, model, ema, epoch, train_steps, checkpoint_dir, by='step')

            if train_steps >= args.max_training_steps:
                logger.info(f"Breaking step loop at train_steps={train_steps}.")
                break

        if train_steps >= args.max_training_steps:
            logger.info(f"Breaking epoch loop at epoch={epoch}.")
            break

        # Finish an epoch
        if args.ckpt_every_n_epoch > 0 and epoch % args.ckpt_every_n_epoch == 0:
            save_checkpoint(args, rank, logger, model, ema, epoch, train_steps, checkpoint_dir, by='epoch')

        epoch += 1

    save_checkpoint(args, rank, logger, model, ema, epoch, train_steps, checkpoint_dir, by='final')

    dist.destroy_process_group()


if __name__ == "__main__":
    # Start
    main(get_args())
