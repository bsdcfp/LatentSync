import os
import sys
import torch


current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录的父目录添加到 sys.path 中
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from convert_torch_to_trt import convert_models
from get_model import get_unet_model, get_vae_decode_model, get_vae_encode_model


if __name__ == "__main__":

    export_onnx = True
    use_plugin = False
    opt_onnx_model = False
    sim_onnx_model = False
    do_classifier_free_guidance = False

    if export_onnx:
        # NOTE: do load model process
        unet_path = "/model_zoo/latentsync_online_simple_auxilary//weights/latentsync/latentsync_unet.pt"
        config_path = "/home/fuping.chu/Avatar/yss_LatentSync/configs/unet/second_stage.yaml"

        model = get_unet_model(config_path, unet_path)

        # vae_path = "stabilityai/sd-vae-ft-mse"
        # model = get_vae_decode_model(vae_path)
        # model = get_vae_encode_model(vae_path)

    else:
        model = None

    output_path = "models/onnx"
    opset = 17
    
    fp16 = True
    
    gpu_path = torch.cuda.get_device_name(torch.cuda.current_device()).split()[-1]
    print("GPU type: ", gpu_path)

    plugin_path = None
    if use_plugin:
        plugin_dir = os.path.join('models/tensorrt', gpu_path)
        if not os.path.exists(plugin_dir):
            os.makedirs(plugin_dir)
        plugin_path = os.path.join('models/tensorrt', gpu_path, 'libplugin.so')
        opt_onnx_model = True
        sim_onnx_model = True

    convert_models(
        model,
        output_path,
        opset,
        fp16,
        unet=True,
        vae_decode=False,
        vae_encode=False,
        clip=False,
        human_matting=False,
        export_onnx=export_onnx,
        max_batch_size=1,
        opt_batch_size=1,
        static_batch=False,
        opt_onnx_model=opt_onnx_model,
        sim_onnx_model=sim_onnx_model,
        plugin_path=plugin_path,
        do_classifier_free_guidance=do_classifier_free_guidance
    )

