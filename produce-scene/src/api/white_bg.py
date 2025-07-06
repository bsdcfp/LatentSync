import numpy as np
from PIL import Image

from ..utils.post_process import paste_fg, paste_fg_sharpen
from ..utils.pre_process import pad_resize_v2


class WhiteBGRunner():
    def __init__(self, gen_h=768, gen_w=768):
        self.gen_h = gen_h
        self.gen_w = gen_w

    def __call__(self, image, mask, 
                 out_h=1024, out_w=1024, bottom=None, template_h=0.8, template_w=0.8, center=None, sharpen=False):
        default_template_cfg = {
            'h': int(out_h*template_h),
            'w': int(out_w*template_w),
            'fg_erode_iter': 0
        }
        if bottom is not None: 
            default_template_cfg['bottom'] = int(out_h * bottom)
        if center is not None:
            default_template_cfg['center'] = (int(out_w*center[0]), int(out_h*center[1]))

        pad_fg, pad_mask, crop_fg_final, crop_mask_final, target_pos = pad_resize_v2(
            image, mask, default_template_cfg, out_h=out_h, out_w=out_w, gen_h=self.gen_h, gen_w=self.gen_w)
        image_gen = Image.fromarray(np.ones_like(pad_fg)*255)

        if sharpen:
            image_addfg = paste_fg_sharpen(np.asarray(image_gen), crop_fg_final, crop_mask_final, target_pos, 
                                           target_h=out_h, target_w=out_w, iter=0)
        else:
            image_addfg = paste_fg(np.asarray(image_gen), crop_fg_final, crop_mask_final, target_pos, 
                                target_h=out_h, target_w=out_w, iter=0)
        
        return {
            'gen_addfg': image_addfg,
            'gen_mask': pad_mask
        }