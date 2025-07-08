# TODO: support dynamic shapes

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录的父目录添加到 sys.path 中
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

import torch
import argparse

from engine import Engine
from model_info import ModelMake
from simandopt.simplify_large_onnx import simplify_large_onnx
from simandopt.fused_gn_silu import opt_graph_with_gn_silu

def call_export_model(in_path, out_path):
    class Args:
        in_model_path = in_path
        out_model_path = out_path
        size_th_kb = 8192
        save_extern_data = 1
        input_shape = ''
        skip = ''

    args = Args()
    simplify_large_onnx(args)

@torch.no_grad()
def convert_models(
    torch_model,
    output_path: str,
    opset: int,
    fp16: bool = False,
    unet: bool = False,
    vae_decode: bool = False,
    vae_encode: bool = False,
    clip: bool = False,
    human_matting: bool = False,
    export_onnx: bool = True,
    max_batch_size: int = 8,
    opt_batch_size: int = 4,
    static_batch: bool = False,
    opt_onnx_model: bool = False,
    sim_onnx_model: bool = False,
    plugin_path: str = None,
    do_classifier_free_guidance: bool = False,
):
    dtype = torch.float16 if fp16 else torch.float32
    if fp16 and torch.cuda.is_available():
        device = "cuda"
        if export_onnx:
            torch_model.cuda()
            torch_model.eval()
    elif fp16 and not torch.cuda.is_available():
        raise ValueError("`float16` model export is only supported on GPUs with CUDA")
    else:
        device = "cpu"
        if export_onnx:
            torch_model.cpu()
            torch_model.eval()

    gpu_path = torch.cuda.get_device_name(torch.cuda.current_device()).split()[-1]

    def engine_path(onnx_path):
        return onnx_path.replace("/onnx/", "/tensorrt/" + gpu_path + "/").replace(
            ".onnx", ".tensorrt"
        )

    # convert torch to tensorrt, fp16 by default
    trt_fp16 = True
    # set to True when convert static batch, otherwise to False
    # do_classifier_free_guidance = True
    re_export_onnx = True

    if unet:
        model_unet = ModelMake()(
            "unet",
            max_batch_size,
            do_classifier_free_guidance,
            device,
            dtype,
            opset,
        )
        if export_onnx:
            unet_model_path = model_unet.export_onnx(
                torch_model, output_path, re_export_onnx
            )
            
            try:
                del torch_model
            except:
                pass
            torch.cuda.empty_cache()
        else:
            unet_model_path = output_path + "/unet/model.onnx"

        if sim_onnx_model:
            ## Step1: simplify large onnx model
            unet_model_sim_path = output_path + "/unet/model.sim.onnx"
            call_export_model(unet_model_path, unet_model_sim_path)
            unet_model_path = unet_model_sim_path

        if opt_onnx_model:
            unet_model_opt_path = output_path + "/unet/model.opt.onnx"
            ## Step2: Opt Model with Plugin
            opt_graph_with_gn_silu(unet_model_path, unet_model_opt_path)
            unet_model_path = unet_model_opt_path

        unet_trt_path = engine_path(unet_model_path)
        model_unet.convert_to_trt(
            unet_model_path,
            unet_trt_path,
            trt_fp16,
            opt_batch_size,
            static_batch=static_batch,
            plugin_path=plugin_path,
        )

    if vae_decode:
        vae_decode = ModelMake()(
            "vae_decode",
            max_batch_size,
            do_classifier_free_guidance,
            device,
            dtype,
            opset,
        )
        if export_onnx:
            vae_decode_path = vae_decode.export_onnx(
                torch_model, output_path, re_export_onnx
            )
            
            try:
                del torch_model
            except:
                pass
            torch.cuda.empty_cache()
        else:
            vae_decode_path = output_path + "/vae_decode/model.onnx"

        if sim_onnx_model:
            ## Step1: simplify large onnx model
            vae_decode_sim_path = output_path + "/vae_decode/model.sim.onnx"
            call_export_model(vae_decode_path, vae_decode_sim_path)
            vae_decode_path = vae_decode_sim_path

        if opt_onnx_model:
            vae_decode_opt_path = output_path + "/vae_decode/model.opt.onnx"
            ## Step2: Opt Model with Plugin
            opt_graph_with_gn_silu(vae_decode_path, vae_decode_opt_path)
            vae_decode_path = vae_decode_opt_path

        vae_decode_trt_path = engine_path(vae_decode_path)
        vae_decode.convert_to_trt(
            vae_decode_path,
            vae_decode_trt_path,
            trt_fp16,
            opt_batch_size,
            static_batch=static_batch,
            plugin_path=plugin_path,
        )

    if vae_encode:
        vae_encode = ModelMake()(
            "vae_encode",
            max_batch_size,
            do_classifier_free_guidance,
            device,
            dtype,
            opset,
        )
        if export_onnx:
            vae_encode_path = vae_encode.export_onnx(
                torch_model, output_path, re_export_onnx
            )
            
            try:
                del torch_model
            except:
                pass
            torch.cuda.empty_cache()
        else:
            vae_encode_path = output_path + "/vae_encode/model.onnx"

        if sim_onnx_model:
            ## Step1: simplify large onnx model
            vae_encode_sim_path = output_path + "/vae_encode/model.sim.onnx"
            call_export_model(vae_encode_path, vae_encode_sim_path)
            vae_encode_path = vae_encode_sim_path

        if opt_onnx_model:
            vae_encode_opt_path = output_path + "/vae_encode/model.opt.onnx"
            ## Step2: Opt Model with Plugin
            opt_graph_with_gn_silu(vae_encode_path, vae_encode_opt_path)
            vae_encode_path = vae_encode_opt_path

        vae_encode_trt_path = engine_path(vae_encode_path)
        vae_encode.convert_to_trt(
            vae_encode_path,
            vae_encode_trt_path,
            trt_fp16,
            opt_batch_size,
            static_batch=static_batch,
            plugin_path=plugin_path,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to the output model."
    )

    parser.add_argument(
        "--opset",
        default=17,
        type=int,
        help="The version of the ONNX operator set to use.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Export the models in `float16` mode",
    )
    parser.add_argument(
        "--unet",
        action="store_true",
        default=False,
        help="Convert unet",
    )
    parser.add_argument(
        "--vae",
        action="store_true",
        default=False,
        help="Convert unet",
    )
    parser.add_argument(
        "--clip",
        action="store_true",
        default=False,
        help="Convert unet",
    )

    args = parser.parse_args()

    model = ""
    # load pytorch model first
    convert_models(
        model, args.output_path, args.opset, args.fp16, args.unet, args.vae, args.clip
    )

