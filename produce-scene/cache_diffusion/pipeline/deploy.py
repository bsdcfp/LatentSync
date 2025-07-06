# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import types
from pathlib import Path

import tensorrt as trt
import torch
from ..cache_diffusion.cachify import CACHED_PIPE, get_model
from cuda import cudart
# from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from tryon.sgm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, TryOnUNetModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from ..pipeline.config import ONNX_CONFIG
# from ..pipeline.models.sd3 import sd3_forward
from ..pipeline.models.sdxl import (
    cachecrossattnupblock2d_forward,
    cacheunet_forward,
    cacheupblock2d_forward,
)
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from torch.onnx import export as onnx_export

from .utils import Engine


def replace_new_forward(backbone):
    if backbone.__class__ == UNet2DConditionModel:
        backbone.forward = types.MethodType(cacheunet_forward, backbone) # cacheunet_forward 绑定到 unet 的 forward
        for upsample_block in backbone.up_blocks:
            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                upsample_block.forward = types.MethodType(
                    cachecrossattnupblock2d_forward, upsample_block
                )
            else:
                upsample_block.forward = types.MethodType(cacheupblock2d_forward, upsample_block)
    # elif backbone.__class__ == SD3Transformer2DModel:
    #     backbone.forward = types.MethodType(sd3_forward, backbone)


def get_input_info(dummy_dict, info: str = None, batch_size: int = 1):
    return_val = [] if info == "profile_shapes" or info == "input_names" or info == "dummy_input" else {}

    def collect_leaf_keys(d):
        for key, value in d.items():
            if isinstance(value, dict):
                collect_leaf_keys(value)
            else:
                value = (value[0] * batch_size,) + value[1:]
                if info == "profile_shapes" and key != "context_20":
                    return_val.append((key, value))  # type: ignore
                elif info == "profile_shapes_dict":
                    return_val[key] = value  # type: ignore
                elif info == "dummy_input":
                    # return_val[key] = torch.ones(value).cuda()  # type: ignore
                    return_val.append(torch.ones(value).cuda())
                elif info == "input_names":
                    return_val.append(key)  # type: ignore

    collect_leaf_keys(dummy_dict)
    if info == "dummy_input":
        return_val = tuple(return_val)
    return return_val

context_size = [
    (2, 320, 96, 72),
    (2, 320, 96, 72),
    (2, 640, 48, 36),
    (2, 640, 48, 36),
    (2, 1280, 24, 18),
    (2, 1280, 24, 18),
    (2, 1280, 12, 9),
    (2, 1280, 12, 9),
    (2, 1280, 12, 9),
    (2, 1280, 12, 9),
    (2, 1280, 12, 9),
    (2, 1280, 24, 18),
    (2, 1280, 24, 18),
    (2, 1280, 24, 18),
    (2, 1280, 48, 36),
    (2, 640, 48, 36),
    (2, 640, 48, 36),
    (2, 640, 96, 72),
    (2, 320, 96, 72),
    (2, 320, 96, 72),
    (2, 320, 96, 72),
]

def get_sample_input(batch_size=1):
    device = "cuda"
    dtype = torch.float32
    xB = 2
    context_input = [
        torch.randn(item).to(device=device, dtype=dtype)
        for item in context_size
    ]
    return (
        torch.randn(
            xB * batch_size, 12, 96, 72
        ).to(device=device, dtype=dtype),
        torch.randn(xB * batch_size, 1280).to(
            device=device, dtype=dtype
        ),
        context_input,
        torch.randn(xB * batch_size, 1, 1280).to(
            device=device, dtype=dtype
        ),
    )

def complie2trt(cls, onnx_path: Path, engine_path: Path, batch_size: int = 1):
    subdirs = [f for f in onnx_path.iterdir() if f.is_dir()]
    for subdir in subdirs:
        if subdir.name not in ONNX_CONFIG[cls].keys():
            continue
        model_path = subdir / "model.onnx"
        plan_path = engine_path / f"{subdir.name}.plan"
        if not plan_path.exists():
            print(f"Building {str(model_path)}")
            build_profile = Profile()
            profile_shapes = get_input_info(
                ONNX_CONFIG[cls][subdir.name]["dummy_input"], "profile_shapes", batch_size
            )
            for input_name, input_shape in profile_shapes:
                min_input_shape = (2,) + input_shape[1:]
                build_profile.add(input_name, min_input_shape, input_shape, input_shape)
            block_network = network_from_onnx_path(
                str(model_path), flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]
            )
            build_config = CreateConfig(
                builder_optimization_level=4,
                profiles=[build_profile],
            )
            engine = engine_from_network(
                block_network,
                config=build_config,
            )
            save_engine(engine, path=plan_path)
        else:
            print(f"{str(model_path)} already exists!")


def get_total_device_memory(backbone):
    max_device_memory = 0
    for _, engine in backbone.engines.items():
        max_device_memory = max(max_device_memory, engine.engine.device_memory_size)
    return max_device_memory


def load_engines(backbone, engine_path: Path, batch_size: int = 1):
    backbone.engines = {}
    print("[INFO] load: ", engine_path)
    for f in engine_path.iterdir():
        if f.is_file():
            print("[INFO] load: ", f"{f.stem}", engine_path)
            eng = Engine()
            eng.load(str(f))
            for name, module in backbone.named_modules():
                if name == f.stem:
                    module.to_empty(device="cpu")
                    torch.cuda.empty_cache()
                    break
            backbone.engines[f"blocks"] = eng
    _, shared_device_memory = cudart.cudaMalloc(get_total_device_memory(backbone))
    for engine in backbone.engines.values():
        engine.activate(shared_device_memory)
    backbone.cuda_stream = cudart.cudaStreamCreate()[1]
    for block_name in backbone.engines.keys():
        backbone.engines[block_name].allocate_buffers(
            shape_dict=get_input_info(
                ONNX_CONFIG[backbone.__class__][block_name]["dummy_input"],
                "profile_shapes_dict",
                batch_size,
            ),
            device="cuda",
            batch_size=batch_size,
        )


def free_memory(model_id, backbone):
    if model_id == "sd3-medium":
        for block in backbone.transformer_blocks:
            block.to_empty(device="cpu")
    else:
        backbone.mid_block.to_empty(device="cpu")
        backbone.down_blocks.to_empty(device="cpu")
        backbone.up_blocks.to_empty(device="cpu")
        torch.cuda.empty_cache()


def export_onnx(backbone, onnx_path: Path):
    for name, module in backbone.named_modules():
        if isinstance(module, CACHED_PIPE[backbone.__class__]):
            _onnx_dir = onnx_path.joinpath(f"{name}")
            _onnx_file = _onnx_dir.joinpath("model.onnx")
            if not _onnx_file.exists():
                _onnx_dir.mkdir(parents=True, exist_ok=True)
                # dummy_input = get_input_info(
                #     ONNX_CONFIG[backbone.__class__][f"{name}"]["dummy_input"], "dummy_input"
                # )
                dummy_input = get_sample_input()
                input_names = get_input_info(
                    ONNX_CONFIG[backbone.__class__][f"{name}"]["dummy_input"], "input_names"
                )
                output_names = ONNX_CONFIG[backbone.__class__][f"{name}"]["output_names"]
                onnx_export(
                    module,
                    args=dummy_input,
                    f=_onnx_file.as_posix(),
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=ONNX_CONFIG[backbone.__class__][f"{name}"]["dynamic_axes"],
                    do_constant_folding=True,
                    opset_version=17,
                )
                onnx_model_path = str(_onnx_file.absolute().as_posix())
                onnx_opt_graph = onnx.load(onnx_model_path)

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
            else:
                print(f"{str(_onnx_file)} alread exists!")


def warm_up(backbone, batch_size: int = 1):
    print("Warming-up TensorRT engines...")
    for name, engine in backbone.engines.items():
        dummy_input = get_input_info(
            ONNX_CONFIG[backbone.__class__][name]["dummy_input"], "dummy_input", batch_size
        )
        _ = engine(dummy_input, backbone.cuda_stream)


def teardown(pipe):
    backbone = get_model(pipe)
    for engine in backbone.engines.values():
        del engine

    cudart.cudaStreamDestroy(backbone.cuda_stream)
    del backbone.cuda_stream


# def compile(pipe, model_id: str, onnx_path: Path, engine_path: Path, batch_size: int = 1):
#     backbone = get_model(pipe)  # get unet
#     onnx_path.mkdir(parents=True, exist_ok=True)
#     engine_path.mkdir(parents=True, exist_ok=True)

#     # replace_new_forward(backbone)
#     export_onnx(backbone, onnx_path)
#     complie2trt(backbone.__class__, onnx_path, engine_path, batch_size)
#     load_engines(backbone, engine_path, batch_size)
#     # free_memory(model_id, backbone)
#     # warm_up(backbone, batch_size)
#     backbone.use_trt_infer = True

def compile(pipe, onnx_path: Path, engine_path: Path, batch_size: int = 1):
    backbone = get_model(pipe)  # get unet
    # print("[INFO] backbone: ", backbone)
    onnx_path.mkdir(parents=True, exist_ok=True)
    engine_path.mkdir(parents=True, exist_ok=True)

    # replace_new_forward(backbone)
    export_onnx(backbone, onnx_path)
    # complie2trt(backbone.__class__, onnx_path, engine_path, batch_size)
    load_engines(backbone, engine_path, batch_size)
    # free_memory(model_id, backbone)
    # warm_up(backbone, batch_size)
    backbone.use_trt_infer = True