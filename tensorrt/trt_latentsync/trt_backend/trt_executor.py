from engine import Engine
from model_info import ModelMake
from cuda import cudart
import os


class TrtExecutor:
    def __init__(
        self,
        stages,
        device="cuda",
        verbose=False,
        nvtx_profile=False,
        use_cuda_graph=False,
    ):
        # initialized in loadResources()
        self.events = {}
        self.models = {}
        self.engine = {}
        self.stream = None

        self.device = device
        self.verbose = verbose
        self.nvtx_profile = nvtx_profile
        self.use_cuda_graph = use_cuda_graph

        # stages
        # self.stages = ['vae_encoder', 'clip', 'unet_encoder', 'unet', 'vae_decoder']
        self.stages = stages

    def loadEngines(
        self,
        engine_dir,
    ):
        plugin_path = engine_dir + "/" + "libplugin.so"
        for stage in self.stages:
            # NOTE: exist opt tensorrt model
            stage_dir = engine_dir + "/" + stage + "/" + "model.opt.tensorrt"
            if os.path.exists(stage_dir):
                self.models[stage] = ModelMake()(stage, True)
                self.engine[stage] = Engine(stage_dir)
                self.engine[stage].load_trt_plugin(plugin_path)
                self.engine[stage].load()
            else:
                stage_dir = engine_dir + "/" + stage + "/" + "model.tensorrt"
                self.models[stage] = ModelMake()(stage, True)
                self.engine[stage] = Engine(stage_dir)
                self.engine[stage].load()

    def loadResources(self):
        # Create CUDA events and stream
        if self.stages is not None:
            for stage in self.stages:
                self.events[stage] = [
                    cudart.cudaEventCreate()[1],
                    cudart.cudaEventCreate()[1],
                ]
        self.stream = cudart.cudaStreamCreate()[1]

    def reshape(self, batch_size=1, remove_cfg=False):
        # allocation TensorRT resources for each stage
        for model_name, obj in self.models.items():
            self.engine[model_name].allocate_buffers(
                shape_dict=obj.get_shape_dict(batch_size, remove_cfg), device=self.device
            )

    def calculateMaxDeviceMemory(self):
        max_device_memory = 0
        for model_name, engine in self.engine.items():
            max_device_memory = max(max_device_memory, engine.engine.device_memory_size)
        return max_device_memory

    def activateEngines(self, shared_device_memory=None):
        if shared_device_memory is None:
            max_device_memory = self.calculateMaxDeviceMemory()
            _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
        self.shared_device_memory = shared_device_memory
        # Load and activate TensorRT engines
        for engine in self.engine.values():
            engine.activate(reuse_device_memory=self.shared_device_memory)

    def runEngine(self, model_name, feed_dict):
        engine = self.engine[model_name]
        return engine.infer(feed_dict, self.stream, use_cuda_graph=self.use_cuda_graph)

    def __call__(self, model_name, feed_dict):
        return self.runEngine(model_name, feed_dict)


if __name__ == "__main__":
    engine_root_path = "/workspace/aigc/tryon_infer/models/tensorrt"
    trt_executor = TrtExecutor()
    trt_executor.loadEngines(engine_root_path)
    _, shared_device_memory = cudart.cudaMalloc(trt_executor.calculateMaxDeviceMemory())
    trt_executor.activateEngines(shared_device_memory)
    trt_executor.loadResources()
