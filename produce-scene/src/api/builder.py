# from .sdxl_brushnet_ipadapter import SDXLBrushNetIpadapterRunner
# from .sdxl_brushnet import SDXLBrushNetRunner
from .white_bg import WhiteBGRunner
from .hunyuan_brushnet import HunyuanBrushNetRunner


scene_image_runner_register = {
    # "SDXLBrushNetIpadapterRunner": SDXLBrushNetIpadapterRunner,
    # "SDXLBrushNetRunner": SDXLBrushNetRunner,
    "WhiteBGRunner": WhiteBGRunner,
    "HunyuanBrushNetRunner": HunyuanBrushNetRunner
}


def build_runner(runner_name, **kwargs):
    return scene_image_runner_register[runner_name](**kwargs)