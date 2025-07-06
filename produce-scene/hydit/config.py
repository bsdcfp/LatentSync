import argparse
import deepspeed

from .constants import *
from .diffusion.gaussian_diffusion import ModelVarType
from .modules.models import HUNYUAN_DIT_CONFIG


def model_var_type(value):
    try:
        return ModelVarType[value]
    except KeyError:
        raise ValueError(f"Invalid choice '{value}', valid choices are {[v.name for v in ModelVarType]}")


def get_args(default_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-flag", type=str)

    # General Setting
    parser.add_argument("--batch-size", type=int, default=1, help="Per-GPU batch size")
    parser.add_argument('--seed', type=int, default=42, help="A seed for all the prompts.")
    parser.add_argument("--use-fp16", action="store_true", help="Use FP16 precision.")
    parser.add_argument("--no-fp16", dest="use_fp16", action="store_false")
    parser.set_defaults(use_fp16=True)
    parser.add_argument("--extra-fp16", action="store_true", help="Use extra fp16 for vae and text_encoder.")

    # HunYuan-DiT
    parser.add_argument("--model", type=str, choices=list(HUNYUAN_DIT_CONFIG.keys()), default='DiT-g/2')
    parser.add_argument("--image-size", type=int, nargs='+', default=[1024, 1024],
                        help='Image size (h, w). If a single value is provided, the image will be treated to '
                             '(value, value).')
    parser.add_argument("--qk-norm", action="store_true",
                        help="Query Key normalization. See http://arxiv.org/abs/2302.05442 for details.")
    parser.set_defaults(qk_norm=True)
    parser.add_argument("--norm", type=str, choices=["rms", "laryer"], default="layer", help="Normalization layer type")
    parser.add_argument("--text-states-dim", type=int, default=1024, help="Hidden size of CLIP text encoder.")
    parser.add_argument("--text-len", type=int, default=77, help="Token length of CLIP text encoder output.")
    parser.add_argument("--text-states-dim-t5", type=int, default=2048, help="Hidden size of CLIP text encoder.")
    parser.add_argument("--text-len-t5", type=int, default=256, help="Token length of T5 text encoder output.")

    # LoRA config
    parser.add_argument("--training-parts", type=str, default='all', choices=['all', 'lora'], help="Training parts")
    parser.add_argument("--rank", type=int, default=64, help="Rank of LoRA")
    parser.add_argument("--lora-ckpt", type=str, default=None, help="LoRA checkpoint")
    parser.add_argument('--target-modules', type=str, nargs='+', default=['Wqkv', 'q_proj', 'kv_proj', 'out_proj'],
                        help="Target modules for LoRA fine tune")
    parser.add_argument("--output-merge-path", type=str, default=None, help="Output path for merged model")

    # controlnet config
    parser.add_argument("--control-type", type=str, default='canny', choices=['canny', 'depth', 'pose'],
                        help="Controlnet condition type")
    parser.add_argument("--control-weight", type=str, default='1.0',
                        help="Controlnet weight, You can use a float to specify the weight for all layers, "
                             "or use a list to separately specify the weight for each layer, for example, "
                             "'[1.0 * (0.825 ** float(19 - i)) for i in range(19)]'")
    parser.add_argument("--condition-image-path", type=str, default=None, help="Inference condition image path")

    # Diffusion
    parser.add_argument("--learn-sigma", action="store_true", help="Learn extra channels for sigma.")
    parser.add_argument("--no-learn-sigma", dest="learn_sigma", action="store_false")
    parser.set_defaults(learn_sigma=True)
    parser.add_argument("--predict-type", type=str, choices=list(PREDICT_TYPE), default="v_prediction",
                        help="Diffusion predict type")
    parser.add_argument("--noise-schedule", type=str, choices=list(NOISE_SCHEDULES), default="scaled_linear",
                        help="Noise schedule")
    parser.add_argument("--beta-start", type=float, default=0.00085, help="Beta start value")
    parser.add_argument("--beta-end", type=float, default=0.02, help="Beta end value")
    parser.add_argument("--sigma-small", action="store_true")
    parser.add_argument("--mse-loss-weight-type", type=str, default="constant",
                        help="Min-SNR-gamma. Can be constant or min_snr_<gamma> where gamma is a integer. "
                             "5 is recommended in the paper.")
    parser.add_argument("--model-var-type", type=model_var_type, default=None, help="Specify the model variable type.")
    parser.add_argument("--noise-offset", type=float, default=0.0, help="Add extra noise to the input image.")

    # ========================================================================================================
    # Inference
    # ========================================================================================================

    # Basic Setting
    parser.add_argument("--prompt", type=str, default="一只小猫", help="The prompt for generating images.")
    parser.add_argument("--model-root", type=str, default="ckpts",
                        help="Root path of all the models, including t2i model and dialoggen model.")
    parser.add_argument("--dit-weight", type=str, default=None,
                        help="Path to the HunYuan-DiT model. If None, search the model in the args.model_root."
                             "1. If it is a file, load the model directly. In this case, the --load-key is ignored."
                             "2. If it is a directory, search the model in the directory. Support two types of models: "
                             "1) named `pytorch_model_*.pt`, where * is specified by the --load-key. "
                             "2) named `*_model_states.pt`, where * can be `mp_rank_00`. *_model_states.pt contains "
                             "both 'module' and 'ema' weights. Therefore, you still use --load-key to specify the "
                             "weights to load. By default, load 'ema' weights. "
                        )
    parser.add_argument("--controlnet-weight", type=str, default=None,
                        help="Path to the HunYuan-DiT controlnet model. If None, search the model in the args.model_root."
                             "1. If it is a directory, search the model in the directory."
                             "2. If it is a file, load the model directly. In this case, the --load-key is ignored."
                        ) 

    # Model setting
    parser.add_argument("--load-key", type=str, choices=["ema", "module", "distill", 'merge'], default="ema",
                        help="Load model key for HunYuanDiT checkpoint.")
    parser.add_argument('--use-style-cond', action="store_true",
                        help="Use style condition in hydit. Only for hydit version <= 1.1")
    parser.add_argument('--size-cond', type=int, nargs='+', default=None,
                        help="Size condition used in sampling. 2 values are required for height and width. "
                             "If a single value is provided, the image will be treated to (value, value)."
                             "Recommended values are [1024, 1024]. Only for hydit version <= 1.1")
    parser.add_argument('--target-ratios', type=str, nargs='+', default=None,
                        help="Target ratios for multi-resolution training.")
    parser.add_argument("--cfg-scale", type=float, default=6.0, help="Guidance scale for classifier-free.")
    parser.add_argument("--negative", type=str, default=None, help="Negative prompt.")

    # Acceleration
    parser.add_argument("--infer-mode", type=str, choices=["fa", "torch", "trt"], default="fa",
                        help="Inference mode")
    parser.add_argument("--onnx-workdir", type=str, default="onnx_model", help="Path to save ONNX model")

    # Sampling
    parser.add_argument("--sampler", type=str, choices=SAMPLER_FACTORY, default="ddpm", help="Diffusion sampler")
    parser.add_argument("--infer-steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--promptfiles", type=str, default='', help="prompt-files")

    # Prompt enhancement
    parser.add_argument("--enhance", action="store_true", help="Enhance prompt with mllm.")
    parser.add_argument("--no-enhance", dest="enhance", action="store_false")
    parser.add_argument("--load-4bit", help="load DialogGen model with 4bit quantization.", action="store_true")
    parser.set_defaults(enhance=True)

    # App
    parser.add_argument("--lang", type=str, default="zh", choices=["zh", "en"], help="Language")


    # ========================================================================================================
    # Training
    # ========================================================================================================

    # Basic Setting
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-training-steps", type=int, default=10_000_000)
    parser.add_argument("--gc-interval", type=int, default=40,
                        help='To address the memory bottleneck encountered during the preprocessing of the dataset,'
                             ' memory fragments are reclaimed here by invoking the gc.collect() function.')
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=100_000, help="Create a ckpt every a few steps.")
    parser.add_argument("--ckpt-latest-every", type=int, default=10_000,
                        help="Create a ckpt named `latest.pt` every a few steps.")
    parser.add_argument("--ckpt-every-n-epoch", type=int, default=0,
                        help="Create a ckpt every a few epochs. If 0, do not create ckpt based on epoch. Default is 0.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=1234)
    parser.add_argument("--warmup-min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-num-steps", type=float, default=0)
    parser.add_argument("--weight-decay", type=float, default=0, help="weight-decay in optimizer")
    parser.add_argument("--rope-img", type=str, default=None, choices=['extend', 'base512', 'base1024'],
                        help="Extend or interpolate the positional embedding of the image.")
    parser.add_argument("--rope-real", action="store_true",
                        help="Use real part and imaginary part separately for RoPE.")

    # Classifier-free
    parser.add_argument("--uncond-p", type=float, default=0.2,
                        help="The probability of dropping training text used for CLIP feature extraction")
    parser.add_argument("--uncond-p-t5", type=float, default=0.2,
                        help="The probability of dropping training text used for mT5 feature extraction")

    # Directory
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-module-root", type=str, default=None, help="Resume model states.")
    parser.add_argument("--resume-ema-root", type=str, default=None, help="Resume ema states.")
    parser.add_argument("--no-strict", dest="strict", action="store_false", help="Strict loading of checkpoint")
    parser.set_defaults(strict=True)

    # Dataset
    parser.add_argument("--index-file", type=str, nargs='+', help="During training, provide a JSON file with data indices.")
    parser.add_argument("--random-flip", action="store_true", help="Random flip image")
    parser.add_argument("--reset-loader", action="store_true",
                        help="Reset the data loader. It is useful when resuming from a checkpoint but switch to a new "
                             "dataset.")
    parser.add_argument("--multireso", action="store_true", help="Use multi-resolution training.")
    parser.add_argument("--reso-step", type=int, default=None, help="Step size for multi-resolution training.")

    # Additional condition
    parser.add_argument("--random-shrink-size-cond", action="store_true",
                        help="Randomly shrink the original size condition.")
    parser.add_argument("--merge-src-cond", action="store_true", help="Merge the source condition into a single value.")

    # EMA Model
    parser.add_argument("--use-ema", action="store_true", help="Use EMA model")
    parser.add_argument("--ema-dtype", type=str, choices=['fp16', 'fp32', 'none'], default="none",
                        help="EMA data type. If none, use the same data type as the model.")
    parser.add_argument("--ema-decay", type=float, default=None,
                        help="EMA decay rate. If None, use the default value of the model.")
    parser.add_argument("--ema-warmup", action="store_true",
                        help="EMA warmup. If True, perform ema_decay warmup from 0 to ema_decay.")
    parser.add_argument("--ema-warmup-power", type=float, default=None,
                        help="EMA power. If None, use the default value of the model.")
    parser.add_argument("--ema-reset-decay", action="store_true",
                        help="Reset EMA decay to 0 and restart increasing the EMA decay."
                             "Only works when --ema-warmup is enabled.")
    # Acceleration
    parser.add_argument("--use-flash-attn", action="store_true", help="During training, "
                                                                      "flash attention is used to accelerate training.")
    parser.add_argument("--no-flash-attn", dest="use_flash_attn", action="store_false",
                        help="During training, flash attention is not used to accelerate training.")
    parser.add_argument("--use-zero-stage", type=int, default=1, help="Use AngelPTM zero stage. Support 2 and 3")
    parser.add_argument("--grad-accu-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Use gradient checkpointing.")
    parser.add_argument("--cpu-offloading", action="store_true", help="Use cpu offloading for parameters and optimizer states.")
    parser.add_argument("--save-optimizer-state", action="store_true", help="Save optimizer state in the checkpoint.")

    # ========================================================================================================
    # Deepspeed config
    # ========================================================================================================
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument('--local_rank', type=int, default=None,
                        help='local rank passed from distributed launcher.')
    parser.add_argument('--deepspeed-optimizer', action='store_true',
                        help='Switching to the optimizers in DeepSpeed')
    parser.add_argument('--remote-device', type=str, default='none', choices=['none', 'cpu', 'nvme'],
                        help='Remote device for ZeRO-3 initialized parameters.')
    parser.add_argument('--zero-stage', type=int, default=1)

    args = parser.parse_args(default_args)

    return args


# convert args to InfeConfig
from typing import List, Optional

class InfeConfig:
    def __init__(
        self,
        task_flag: Optional[str] = None,
        batch_size: int = 1,
        seed: int = 42,
        use_fp16: bool = True,
        extra_fp16: bool = False,
        model: str = 'DiT-g/2',
        image_size: List[int] = [1024, 1024],
        qk_norm: bool = True,
        norm: str = "layer",
        text_states_dim: int = 1024,
        text_len: int = 77,
        text_states_dim_t5: int = 2048,
        text_len_t5: int = 256,
        training_parts: str = 'all',
        rank: int = 64,
        lora_ckpt: Optional[str] = None,
        target_modules: List[str] = ['Wqkv', 'q_proj', 'kv_proj', 'out_proj'],
        output_merge_path: Optional[str] = None,
        control_type: str = 'canny',
        control_weight: str = '1.0',
        condition_image_path: Optional[str] = None,
        learn_sigma: bool = True,
        predict_type: str = "v_prediction",
        noise_schedule: str = "scaled_linear",
        beta_start: float = 0.00085,
        beta_end: float = 0.02,
        sigma_small: bool = False,
        mse_loss_weight_type: str = "constant",
        model_var_type: Optional[str] = None,
        noise_offset: float = 0.0,
        prompt: str = "一只小猫",
        model_root: str = "ckpts",
        dit_weight: Optional[str] = None,
        controlnet_weight: Optional[str] = None,
        load_key: str = "ema",
        use_style_cond: bool = False,
        size_cond: Optional[List[int]] = None,
        target_ratios: Optional[List[str]] = None,
        cfg_scale: float = 6.0,
        negative: Optional[str] = None,
        infer_mode: str = "fa",
        onnx_workdir: str = "onnx_model",
        sampler: str = "ddpm",
        infer_steps: int = 20,
        enhance: bool = True,
        load_4bit: bool = False,
        lang: str = "zh",
        lr: float = 1e-4,
        epochs: int = 100,
        max_training_steps: int = 10_000_000,
        gc_interval: int = 40,
        log_every: int = 100,
        ckpt_every: int = 100_000,
        ckpt_latest_every: int = 10_000,
        ckpt_every_n_epoch: int = 0,
        num_workers: int = 4,
        global_seed: int = 1234,
        warmup_min_lr: float = 1e-6,
        warmup_num_steps: int = 0,
        weight_decay: float = 0,
        rope_img: Optional[str] = None,
        rope_real: bool = False,
        uncond_p: float = 0.2,
        uncond_p_t5: float = 0.2,
        results_dir: str = "results",
        resume: bool = False,
        resume_module_root: Optional[str] = None,
        resume_ema_root: Optional[str] = None,
        strict: bool = True,
        index_file: Optional[List[str]] = None,
        random_flip: bool = False,
        reset_loader: bool = False,
        multireso: bool = False,
        reso_step: Optional[int] = None,
        random_shrink_size_cond: bool = False,
        merge_src_cond: bool = False,
        use_ema: bool = False,
        ema_dtype: str = "none",
        ema_decay: Optional[float] = None,
        ema_warmup: bool = False,
        ema_warmup_power: Optional[float] = None,
        ema_reset_decay: bool = False,
        use_flash_attn: bool = False,
        use_zero_stage: int = 1,
        grad_accu_steps: int = 1,
        gradient_checkpointing: bool = False,
        cpu_offloading: bool = False,
        save_optimizer_state: bool = False,
        local_rank: Optional[int] = None,
        deepspeed_optimizer: bool = False,
        remote_device: str = 'none',
        zero_stage: int = 1,
        enable_teacache: bool = False
    ):
        self.task_flag = task_flag
        self.batch_size = batch_size
        self.seed = seed
        self.use_fp16 = use_fp16
        self.extra_fp16 = extra_fp16
        self.model = model
        self.image_size = image_size
        self.qk_norm = qk_norm
        self.norm = norm
        self.text_states_dim = text_states_dim
        self.text_len = text_len
        self.text_states_dim_t5 = text_states_dim_t5
        self.text_len_t5 = text_len_t5
        self.training_parts = training_parts
        self.rank = rank
        self.lora_ckpt = lora_ckpt
        self.target_modules = target_modules
        self.output_merge_path = output_merge_path
        self.control_type = control_type
        self.control_weight = control_weight
        self.condition_image_path = condition_image_path
        self.learn_sigma = learn_sigma
        self.predict_type = predict_type
        self.noise_schedule = noise_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.sigma_small = sigma_small
        self.mse_loss_weight_type = mse_loss_weight_type
        self.model_var_type = model_var_type
        self.noise_offset = noise_offset
        self.prompt = prompt
        self.model_root = model_root
        self.dit_weight = dit_weight
        self.controlnet_weight = controlnet_weight
        self.load_key = load_key
        self.use_style_cond = use_style_cond
        self.size_cond = size_cond
        self.target_ratios = target_ratios
        self.cfg_scale = cfg_scale
        self.negative = negative
        self.infer_mode = infer_mode
        self.onnx_workdir = onnx_workdir
        self.sampler = sampler
        self.infer_steps = infer_steps
        self.enhance = enhance
        self.load_4bit = load_4bit
        self.lang = lang
        self.lr = lr
        self.epochs = epochs
        self.max_training_steps = max_training_steps
        self.gc_interval = gc_interval
        self.log_every = log_every
        self.ckpt_every = ckpt_every
        self.ckpt_latest_every = ckpt_latest_every
        self.ckpt_every_n_epoch = ckpt_every_n_epoch
        self.num_workers = num_workers
        self.global_seed = global_seed
        self.warmup_min_lr = warmup_min_lr
        self.warmup_num_steps = warmup_num_steps
        self.weight_decay = weight_decay
        self.rope_img = rope_img
        self.rope_real = rope_real
        self.uncond_p = uncond_p
        self.uncond_p_t5 = uncond_p_t5
        self.results_dir = results_dir
        self.resume = resume
        self.resume_module_root = resume_module_root
        self.resume_ema_root = resume_ema_root
        self.strict = strict
        self.index_file = index_file
        self.random_flip = random_flip
        self.reset_loader = reset_loader
        self.multireso = multireso
        self.reso_step = reso_step
        self.random_shrink_size_cond = random_shrink_size_cond
        self.merge_src_cond = merge_src_cond
        self.use_ema = use_ema
        self.ema_dtype = ema_dtype
        self.ema_decay = ema_decay
        self.ema_warmup = ema_warmup
        self.ema_warmup_power = ema_warmup_power
        self.ema_reset_decay = ema_reset_decay
        self.use_flash_attn = use_flash_attn
        self.use_zero_stage = use_zero_stage
        self.grad_accu_steps = grad_accu_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.cpu_offloading = cpu_offloading
        self.save_optimizer_state = save_optimizer_state
        self.local_rank = local_rank
        self.deepspeed_optimizer = deepspeed_optimizer
        self.remote_device = remote_device
        self.zero_stage = zero_stage
        self.enable_teacache = enable_teacache