from .cvcore import main_rect


def get_cropbox_center(mask, crop_tags, target_fg_size=0.9, target_image_ratio=1):
    fg_w, fg_h = main_rect(mask)[2:]
    fg_ratio = fg_h / fg_w
    
    # 特殊case: 对边贴边, 强制调整最大前景占比为1
    if 'top' in crop_tags and 'bottom' in crop_tags and 'left' in crop_tags and 'right' in crop_tags:
        return [0.5, 0.5, 0.5, 0.5]
    if 'top' in crop_tags and 'bottom' in crop_tags:
        if fg_ratio < 1: return [0, 0, 0, 0]
        target_fg_size = 1
    if 'left' in crop_tags and 'right' in crop_tags:
        if fg_ratio > 1: return [0, 0, 0, 0]
        target_fg_size = 1

    max_center_x, min_center_x, max_center_y, min_center_y = 0.5, 0.5, 0.5, 0.5

    if 'left' in crop_tags and 'right' not in crop_tags:
        # 左贴边
        max_w = target_fg_size / fg_ratio if fg_ratio > target_image_ratio else target_fg_size
        max_center_x = max_w / 2
    if 'right' in crop_tags and 'left' not in crop_tags:
        # 右贴边
        max_w = target_fg_size / fg_ratio if fg_ratio > target_image_ratio else target_fg_size
        min_center_x = 1 - max_w / 2

    if 'top' in crop_tags and 'bottom' not in crop_tags:
        # 上贴边
        max_h = target_fg_size * fg_ratio if fg_ratio < target_image_ratio else target_fg_size
        max_center_y = max_h / 2
    if 'bottom' in crop_tags and 'top' not in crop_tags:
        # 下贴边
        max_h = target_fg_size * fg_ratio if fg_ratio < target_image_ratio else target_fg_size
        min_center_y = 1 - max_h / 2

    return max_center_x, min_center_x, max_center_y, min_center_y