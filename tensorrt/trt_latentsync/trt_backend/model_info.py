# TODO: support dynamic image shapes
import torch
from pathlib import Path
import onnx
import os
import shutil

from optimize import Optimizer
from engine import Engine


class BaseModel:
    def __init__(
        self, max_batch_size=16, device="cuda", dtype=torch.float16, onnx_opset=17
    ):
        self.name = self.__class__.__name__
        self.min_batch = 1
        self.max_batch = max_batch_size
        self.device = device
        self.dtype = dtype
        self.onnx_opset = onnx_opset
        self.subfolder = ""
        self.verbose = True


    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size=1):
        return None

    def get_input_profile(self, batch_size, static_batch):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    def get_minmax_dims(self, batch_size, static_batch):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        # latent_height = image_height // 8
        # latent_width = image_width // 8
        # min_image_height = image_height if static_shape else self.min_image_shape
        # max_image_height = image_height if static_shape else self.max_image_shape
        # min_image_width = image_width if static_shape else self.min_image_shape
        # max_image_width = image_width if static_shape else self.max_image_shape
        # min_latent_height = latent_height if static_shape else self.min_latent_shape
        # max_latent_height = latent_height if static_shape else self.max_latent_shape
        # min_latent_width = latent_width if static_shape else self.min_latent_shape
        # max_latent_width = latent_width if static_shape else self.max_latent_shape
        return (min_batch, max_batch)

    def optimize(self, onnx_graph, return_onnx=True, **kwargs):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ": original")
        opt.cleanup()
        opt.info(self.name + ": cleanup")
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        if kwargs.get("fuse_mha_qkv_int8", False):
            opt.fuse_mha_qkv_int8_sq()
            opt.info(self.name + ": fuse QKV nodes")
        onnx_opt_graph = opt.cleanup(return_onnx=return_onnx)
        opt.info(self.name + ": finished")
        return onnx_opt_graph

    def export_onnx(self, model, output_path: str, reexport=False):
        output_path = Path(output_path)
        onnx_path = output_path / self.subfolder / "model.onnx"
        if not os.path.exists(onnx_path.as_posix()) or reexport == True:
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Exporting ONNX model: {onnx_path.as_posix()}")
            with torch.inference_mode(), torch.autocast("cuda"):
                torch.onnx.export(
                    model,
                    self.get_sample_input(),
                    onnx_path.as_posix(),
                    export_params=True,
                    opset_version=self.onnx_opset,
                    do_constant_folding=True,
                    input_names=self.get_input_names(),
                    output_names=self.get_output_names(),
                    dynamic_axes=self.get_dynamic_axes(),
                )

            print(f"[INFO] Optimize ONNX model: {onnx_path.as_posix()}")
            onnx_model_path = str(onnx_path.absolute().as_posix())
            onnx_opt_graph = self.optimize(onnx.load(onnx_model_path))

            if onnx_opt_graph.ByteSize() > 2147483648:

                onnx_dir = os.path.dirname(onnx_model_path)
                # clean up existing tensor files
                shutil.rmtree(onnx_dir)
                os.mkdir(onnx_dir)

                onnx.save_model(
                    onnx_opt_graph,
                    onnx_model_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="weights.pb",
                    convert_attribute=False,
                )
            else:
                onnx.save(onnx_opt_graph, onnx_path)

        return str(onnx_path)

    def convert_to_trt(self, onnx_path, engine_path, fp16, batch_size, static_batch, plugin_path=None):
        engine = Engine(engine_path)
        input_profile = self.get_input_profile(
            batch_size=batch_size, static_batch=static_batch
        )
        if plugin_path:
            engine.load_trt_plugin(plugin_path=plugin_path)
        engine.build(
            onnx_path, fp16=fp16, tf32=False, int8=False, input_profile=input_profile
        )


# _compute_unet_blocks forward called, print shape:
# sample shape: torch.Size([1, 320, 16, 32, 32])
# emb shape: torch.Size([1, 1280])
# encoder_hidden_states shape: torch.Size([16, 50, 384])
# down_block_additional_residuals is None
# mid_block_additional_residual is None
# forward_upsample_size: False

# sample shape: torch.Size([1, 320, 16, 32, 32])

class UnetModel(BaseModel):
    def __init__(
        self,
        max_batch_size=16,
        do_classifier_free_guidance=False,
        device="cuda",
        dtype=torch.float16,
        onnx_opset=17,
    ):
        super(UnetModel, self).__init__(max_batch_size, device, dtype, onnx_opset)
        self.subfolder = "unet"

        # sample
        self.length = 320
        self.channels = 16
        self.height = 32
        self.width = 32

        # emb
        self.emb_length = 1280

        # encoder_hidden_states
        self.encoder_hidden_states_height = 50
        self.encoder_hidden_states_width = 384

        self.xB = 2 if do_classifier_free_guidance else 1  # batch multiplier

    def get_input_names(self):
        return (
            ["sample", "emb", "encoder_hidden_states"]
        )

    def get_output_names(self):
        return ["out_sample"]

    def get_sample_input(self, batch_size=1):
        device = self.device
        dtype = self.dtype
        return (
            torch.randn(
                self.xB * batch_size, self.length, self.channels, self.height, self.width
            ).to(device=device, dtype=dtype), # sample
            torch.randn(self.xB * batch_size, self.emb_length).to(device=device, dtype=dtype), # emb
            torch.randn(self.channels, self.encoder_hidden_states_height, self.encoder_hidden_states_width).to(
                device=device, dtype=dtype
            )
        )

    def get_dynamic_axes(self):
        xB = "2B" if self.xB == 2 else "B"
        return {
            "sample": {0: xB},
            "emb": {0: xB},
            "out_sample": {0: xB},
        }

    # TODO: support dynamic image shape
    def get_shape_dict(self, batch_size=1, remove_cfg=False):
        # dynamic batch size
        self.xB = 1 
        return {
            "sample": (self.xB * batch_size, self.length, self.channels, self.height, self.width),
            "emb": (self.xB * batch_size, self.emb_length),
            "encoder_hidden_states": (self.channels, self.encoder_hidden_states_height, self.encoder_hidden_states_width),
            "out_sample": (
                self.xB * batch_size,
                self.length,
                self.channels,
                self.height,
                self.width,
            ),  # must add output
        }

    # NOTE: profile input for building trt engine, [min, opt, max]
    def get_input_profile(self, batch_size, static_batch):
        min_batch, max_batch = self.get_minmax_dims(
            batch_size=batch_size, static_batch=static_batch
        )
        return {
            "sample": [
                (self.xB * min_batch, self.length, self.channels, self.height, self.width),
                (self.xB * batch_size, self.length, self.channels, self.height, self.width),
                (self.xB * max_batch, self.length, self.channels, self.height, self.width),
            ],
            "emb": [
                (self.xB * min_batch, self.emb_length),
                (self.xB * batch_size, self.emb_length),
                (self.xB * max_batch, self.emb_length),
            ],
            "encoder_hidden_states": [
                (self.channels, self.encoder_hidden_states_height, self.encoder_hidden_states_width),
                (self.channels, self.encoder_hidden_states_height, self.encoder_hidden_states_width),
                (self.channels, self.encoder_hidden_states_height, self.encoder_hidden_states_width),
            ],
        }

class UnetCacheModel(BaseModel):
    def __init__(
        self,
        max_batch_size=16,
        do_classifier_free_guidance=True,
        device="cuda",
        dtype=torch.float16,
        onnx_opset=17,
    ):
        super(UnetCacheModel, self).__init__(max_batch_size, device, dtype, onnx_opset)
        self.subfolder = "unet_cache"

        # sample
        self.channels = 12
        self.height = 128
        self.width = 96
        self.clip_length = 1280
        self.xB = 2 if do_classifier_free_guidance else 1  # batch multiplier
        self.output_channels = 4

    def get_input_names(self):
        return (
            ["hidden_states", "temb"]
            + ["context_%d" % i for i in range(len(self.cache_context_size))]
            + ["clip_context"]
        )

    def get_output_names(self):
        return ["hidden_states_out"]

    def get_sample_input(self, batch_size=1):
        device = self.device
        dtype = self.dtype
        context_input = [
            torch.randn(item).to(device=device, dtype=dtype)
            for item in self.cache_context_size
        ]
        return (
            torch.randn(
                2 * batch_size, self.channels, self.height, self.width
            ).to(device=device, dtype=dtype),
            torch.randn(2 * batch_size, self.clip_length).to(
                device=device, dtype=dtype
            ),
            context_input,
            torch.randn(2 * batch_size, 1, self.clip_length).to(
                device=device, dtype=dtype
            ),
        )

    def get_dynamic_axes(self):
        xB = "2B" if self.xB == 2 else "B"
        return {
            "hidden_states": {0: xB},
            "temb": {0: xB},
            **{"context_%d" % i: {0: xB} for i in range(len(self.cache_context_size))},
            "clip_context": {0: xB},
            "hidden_states_out": {0: xB},
        }

    # TODO: support dynamic image shape
    def get_shape_dict(self, batch_size=1, remove_cfg=False):
        # dynamic batch size
        self.xB = 1 if remove_cfg else 2 # reset batch size 
        return {
            "hidden_states": (self.xB * batch_size, self.channels, self.height, self.width),
            "temb": (self.xB * batch_size, self.clip_length),
            **{
                f"context_{i}": (self.xB * batch_size, *item[1:])
                for i, item in enumerate(self.cache_context_size)
            },
            "clip_context": (self.xB * batch_size, 1, self.clip_length),
            "hidden_states_out": (
                self.xB * batch_size,
                640,
                128,
                96,
            ),  # must add output
        }

    # NOTE: profile input for building trt engine, [min, opt, max]
    def get_input_profile(self, batch_size, static_batch):
        min_batch, max_batch = self.get_minmax_dims(
            batch_size=batch_size, static_batch=static_batch
        )
        return {
            "hidden_states": [
                (self.xB * min_batch, self.channels, self.height, self.width),
                (self.xB * batch_size, self.channels, self.height, self.width),
                (self.xB * max_batch, self.channels, self.height, self.width),
            ],
            "temb": [
                (self.xB * min_batch, self.clip_length),  # Must use list for one dim input
                (self.xB * batch_size, self.clip_length),
                (self.xB * max_batch, self.clip_length),
            ],
            **{
                f"context_{i}": [
                    (self.xB * min_batch, *item[1:]),
                    (self.xB * batch_size, *item[1:]),
                    (self.xB * max_batch, *item[1:]),
                ]
                for i, item in enumerate(self.cache_context_size)
            },
            "clip_context": [
                (self.xB * min_batch, 1, self.clip_length),
                (self.xB * batch_size, 1, self.clip_length),
                (self.xB * max_batch, 1, self.clip_length),
            ],
        }

class UnetLastModel(BaseModel):
    def __init__(
        self,
        max_batch_size=16,
        do_classifier_free_guidance=True,
        device="cuda",
        dtype=torch.float16,
        onnx_opset=17,
    ):
        super(UnetLastModel, self).__init__(
            max_batch_size, device, dtype, onnx_opset
        )
        self.subfolder = "unet_last"

        # sample
        self.channels = 640
        self.height = 128
        self.width = 96
        self.clip_length = 1280
        self.xB = 2 if do_classifier_free_guidance else 1  # batch multiplier

    def get_input_names(self):
        return ["hidden_states", "temb", "context", "clip_context"]

    def get_output_names(self):
        return ["hidden_states_out"]

    def get_dynamic_axes(self):
        xB = "2B" if self.xB == 2 else "B"
        return {
            "hidden_states": {0: xB},
            "temb": {0: xB},
            "context": {0: xB},
            "clip_context": {0: xB},
            "hidden_states_out": {0: xB},
        }

    def get_sample_input(self, batch_size=1):
        device = self.device
        dtype = self.dtype
        return (
            torch.randn(
                2 * batch_size, self.channels, self.height, self.width
            ).to(device=device, dtype=dtype),
            torch.randn(2 * batch_size, self.clip_length).to(device=device, dtype=dtype),
            torch.randn(2 * batch_size, 320, self.height, self.width).to(
                device=device, dtype=dtype
            ),
            torch.randn(2 * batch_size, 1, self.clip_length).to(
                device=device, dtype=dtype
            ),
        )

    def get_shape_dict(self, batch_size=1, remove_cfg=False):
        # dynamic batch size
        self.xB = 1 if remove_cfg else 2 # reset batch size 
        param = {
            "hidden_states": (
                self.xB * batch_size,
                self.channels,
                self.height,
                self.width,
            ),
            "temb": (self.xB * batch_size, self.clip_length),
            "context": (self.xB * batch_size, 320, self.height, self.width),
            "clip_context": (self.xB * batch_size, 1, self.clip_length),
            "hidden_states_out": (
                self.xB * batch_size,
                4,
                128,
                96,
            ),  # must add output
        }

        return param

    # NOTE: profile input for build trt engine, [min, opt, max]
    def get_input_profile(self, batch_size, static_batch):
        min_batch, max_batch = self.get_minmax_dims(
            batch_size=batch_size, static_batch=static_batch
        )
        return {
            "hidden_states": [
                (self.xB * min_batch, self.channels, self.height, self.width),
                (self.xB * batch_size, self.channels, self.height, self.width),
                (self.xB * max_batch, self.channels, self.height, self.width),
            ],
            "temb": [
                (self.xB * min_batch, self.clip_length),  # Must use list for one dim input
                (self.xB * batch_size, self.clip_length),
                (self.xB * max_batch, self.clip_length),
            ],
            "context": [
                (self.xB * min_batch, 320, self.height, self.width),
                (self.xB * batch_size, 320, self.height, self.width),
                (self.xB * max_batch, 320, self.height, self.width),
            ],
            "clip_context": [
                (self.xB * min_batch, 1, self.clip_length),
                (self.xB * batch_size, 1, self.clip_length),
                (self.xB * max_batch, 1, self.clip_length),
            ],
        }

#vae decode shape torch.Size([16, 4, 32, 32]) torch.Size([16, 3, 256, 256])
class VaeDecodeModel(BaseModel):
    def __init__(
        self,
        max_batch_size=16,
        do_classifier_free_guidance=False,
        device="cuda",
        dtype=torch.float16,
        onnx_opset=17,
    ):
        super(VaeDecodeModel, self).__init__(
            max_batch_size, device, dtype, onnx_opset
        )
        self.subfolder = "vae_decode"

        # sample
        self.input_channels = 4
        self.input_height = 32
        self.input_width = 32
        self.input_length = 16

        self.output_channels = 3
        self.output_height = 256
        self.output_width = 256

        self.xB = 2 if do_classifier_free_guidance else 1  # batch multiplier

    def get_input_names(self):
        return ["latents", ]

    def get_output_names(self):
        return ["output"]

    def get_dynamic_axes(self):
        xB = "2B" if self.xB == 2 else "B"
        return {
            "latents": {0: xB},
        }

    def get_sample_input(self, batch_size=1):
        device = self.device
        dtype = self.dtype
        return (
            torch.randn(
                batch_size * self.input_length, self.input_channels, self.input_height, self.input_width
            ).to(device=device, dtype=dtype),
        )

    def get_shape_dict(self, batch_size=1, remove_cfg=False):
        # dynamic batch size
        self.xB = 1
        param = {
            "latents": (
                self.xB * batch_size * self.input_length,
                self.input_channels,
                self.input_height,
                self.input_width,
            ),
            "output": (
                self.xB * batch_size * self.input_length,
                self.output_channels,
                self.output_height,
                self.output_width,
            ),  # must add output
        }

        return param

    # NOTE: profile input for build trt engine, [min, opt, max]
    def get_input_profile(self, batch_size, static_batch):
        min_batch, max_batch = self.get_minmax_dims(
            batch_size=batch_size, static_batch=static_batch
        )
        return {
            "latents": [
                (self.xB * min_batch * self.input_length, self.input_channels, self.input_height, self.input_width),
                (self.xB * batch_size * self.input_length, self.input_channels, self.input_height, self.input_width),
                (self.xB * max_batch * self.input_length, self.input_channels, self.input_height, self.input_width),
            ],
        }


# images: torch.Size([16, 3, 256, 256])
# image_latents: torch.Size([16, 4, 32, 32])
class VaeEncodeModel(BaseModel):
    def __init__(
        self,
        max_batch_size=16,
        do_classifier_free_guidance=False,
        device="cuda",
        dtype=torch.float16,
        onnx_opset=17,
    ):
        super(VaeEncodeModel, self).__init__(
            max_batch_size, device, dtype, onnx_opset
        )
        self.subfolder = "vae_encode"

        # sample
        self.input_channels = 3
        self.input_height = 256
        self.input_width = 256
        self.input_length = 16

        self.output_channels = 4
        self.output_height = 32
        self.output_width = 32

        self.xB = 2 if do_classifier_free_guidance else 1  # batch multiplier

    def get_input_names(self):
        return ["latents", ]

    def get_output_names(self):
        return ["output"]

    def get_dynamic_axes(self):
        xB = "2B" if self.xB == 2 else "B"
        return {
            "latents": {0: xB},
        }

    def get_sample_input(self, batch_size=1):
        device = self.device
        dtype = self.dtype
        return (
            torch.randn(
                batch_size * self.input_length, self.input_channels, self.input_height, self.input_width
            ).to(device=device, dtype=dtype),
        )

    def get_shape_dict(self, batch_size=1, remove_cfg=False):
        # dynamic batch size
        self.xB = 1
        param = {
            "latents": (
                self.xB * batch_size * self.input_length,
                self.input_channels,
                self.input_height,
                self.input_width,
            ),
            "output": (
                self.xB * batch_size * self.input_length,
                self.output_channels,
                self.output_height,
                self.output_width,
            ),  # must add output
        }

        return param

    # NOTE: profile input for build trt engine, [min, opt, max]
    def get_input_profile(self, batch_size, static_batch):
        min_batch, max_batch = self.get_minmax_dims(
            batch_size=batch_size, static_batch=static_batch
        )
        return {
            "latents": [
                (self.xB * min_batch * self.input_length, self.input_channels, self.input_height, self.input_width),
                (self.xB * batch_size * self.input_length, self.input_channels, self.input_height, self.input_width),
                (self.xB * max_batch * self.input_length, self.input_channels, self.input_height, self.input_width),
            ],
        }


class ModelMake:
    def __call__(
        self,
        stage,
        max_batch_size=16,
        do_classifier_free_guidance=True,
        device="cuda",
        dtype=torch.float16,
        onnx_opset=17,
    ):
        if stage == "unet":
            return UnetModel(
                max_batch_size, do_classifier_free_guidance, device, dtype, onnx_opset
            )
        elif stage == "unet_cache":
            return UnetCacheModel(max_batch_size, do_classifier_free_guidance, device, dtype, onnx_opset)
        elif stage == "unet_last":
            return UnetLastModel(max_batch_size, do_classifier_free_guidance, device, dtype, onnx_opset)
        elif stage == "vae_decode":
            return VaeDecodeModel(max_batch_size, do_classifier_free_guidance, device, dtype, onnx_opset)
        elif stage == "vae_encode":
            return VaeEncodeModel(max_batch_size, do_classifier_free_guidance, device, dtype, onnx_opset)
        else:
            raise ValueError(f"Invalid stage: {stage}")
