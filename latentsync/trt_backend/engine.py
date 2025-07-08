from collections import OrderedDict
from cuda import cudart
import numpy as np
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt.loader import LoadPlugins
from polygraphy.backend.trt import (
    CreateConfig,
    ModifyNetworkOutputs,
    Profile,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)

import tensorrt as trt
import torch
import os

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
}


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None


class Engine:
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None  # cuda graph

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def refit(self, refit_weights, is_fp16):
        # Initialize refitter
        refitter = trt.Refitter(self.engine, TRT_LOGGER)

        refitted_weights = set()
        # iterate through all tensorrt refittable weights
        for trt_weight_name in refitter.get_all_weights():
            if trt_weight_name not in refit_weights:
                continue

            # get weight from state dict
            trt_datatype = trt.DataType.FLOAT
            if is_fp16:
                refit_weights[trt_weight_name] = refit_weights[trt_weight_name].half()
                trt_datatype = trt.DataType.HALF

            # trt.Weight and trt.TensorLocation
            trt_wt_tensor = trt.Weights(
                trt_datatype,
                refit_weights[trt_weight_name].data_ptr(),
                torch.numel(refit_weights[trt_weight_name]),
            )
            trt_wt_location = (
                trt.TensorLocation.DEVICE
                if refit_weights[trt_weight_name].is_cuda
                else trt.TensorLocation.HOST
            )

            # apply refit
            refitter.set_named_weights(trt_weight_name, trt_wt_tensor, trt_wt_location)
            refitted_weights.add(trt_weight_name)

        assert set(refitted_weights) == set(refit_weights.keys())
        if not refitter.refit_cuda_engine():
            print("Error: failed to refit new weights.")
            exit(0)

        print(f"[I] Total refitted weights {len(refitted_weights)}.")

    def build(
        self,
        onnx_path,
        strongly_typed=False,
        fp16=True,
        bf16=False,
        tf32=False,
        int8=False,
        input_profile=None,
        enable_refit=False,
        enable_all_tactics=False,
        timing_cache=None,
        update_output_names=None,
        native_instancenorm=True,
        **extra_build_args,
    ):
        print(f"[INFO] Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        if not enable_all_tactics:
            extra_build_args["tactic_sources"] = []

        flags = []
        if native_instancenorm:
            flags.append(trt.OnnxParserFlag.NATIVE_INSTANCENORM)
        network = network_from_onnx_path(
            onnx_path,
            flags=flags,
        )

        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)
        engine = engine_from_network(
            network,
            config=CreateConfig(
                fp16=fp16,
                # bf16=bf16,
                tf32=tf32,
                int8=int8,
                refittable=enable_refit,
                profiles=[p],
                load_timing_cache=timing_cache,
                # memory_pool_limits={trt.MemoryPoolType.WORKSPACE:16 * 1024 * 1024 * 1024},
                **extra_build_args,
            ),
            save_timing_cache=timing_cache,
        )
        save_engine(engine, path=self.engine_path)

    def load(self):
        print(f"[INFO] Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def load_trt_plugin(self, plugin_path):
        if not os.path.exists(plugin_path):
            raise ValueError("Trt plugin {path} not exist! Please check".format(path = plugin_path))
        print(f"[INFO] Loading TensorRT plugins: {plugin_path}")
        LoadPlugins(plugins=[plugin_path])()

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device=device)
            self.tensors[binding] = tensor

    def infer(self, feed_dict, stream, use_cuda_graph=False):

        for name, buf in feed_dict.items():
            # print("--------", self.tensors[name].shape, buf.shape)
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream))
                CUASSERT(cudart.cudaStreamSynchronize(stream))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream)
                if not noerror:
                    raise ValueError(f"ERROR: inference failed.")
                # capture cuda graph
                CUASSERT(
                    cudart.cudaStreamBeginCapture(
                        stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal
                    )
                )
                self.context.execute_async_v3(stream)
                self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream))
                self.cuda_graph_instance = CUASSERT(
                    cudart.cudaGraphInstantiate(self.graph, 0)
                )
        else:
            noerror = self.context.execute_async_v3(stream)
            if not noerror:
                raise ValueError(f"ERROR: inference failed.")

        return self.tensors
