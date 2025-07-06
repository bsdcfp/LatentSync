import cv2

from .cvcore import pad_and_erode
from .sharpen import img_sharpening_v2


def paste_fg(image_gen, fg, mask, target_pos, target_w=1080, target_h=1080, iter=1):
    """
    args:
        image_gen: hxwx3, rgb
        fg: hxwx3, rgb, 
        mask: hxwx3
        target_pos: x, y, w, h
        target_w, target_h: output image size
    """
    mask = pad_and_erode(mask, iter=iter)
    x, y, w, h = target_pos
    if target_w == image_gen.shape[1] and target_h == image_gen.shape[0]:
        out = image_gen.copy()
    else:
        out = cv2.resize(image_gen, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    out[y:(y+h), x:(x+w)] = out[y:(y+h), x:(x+w)] * (1 - mask/255.) + fg * (mask/255.)
    return out


def paste_fg_sharpen(image_gen, fg, mask, target_pos, target_w=1080, target_h=1080, iter=1):
    """
    args:
        image_gen: hxwx3, rgb
        fg: hxwx3, rgb, 
        mask: hxwx3
        target_pos: x, y, w, h
        target_w, target_h: output image size
    """
    mask = pad_and_erode(mask, iter=iter)
    x, y, w, h = target_pos
    if target_w == image_gen.shape[1] and target_h == image_gen.shape[0]:
        out = image_gen.copy()
    else:
        out = cv2.resize(image_gen, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    # sharpen fg
    fg = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
    fg = img_sharpening_v2(fg)
    fg = cv2.cvtColor(fg, cv2.COLOR_RGB2BGR)
    
    out[y:(y+h), x:(x+w)] = out[y:(y+h), x:(x+w)] * (1 - mask/255.) + fg * (mask/255.)
    return out