import matplotlib.pyplot as plt
import math


def vis_image_list(images, save_to):
    num = len(images)
    num_row = 1 if num < 3 else math.ceil(num / 3)
    num_col = num if num < 3 else 3
    fig, axes = plt.subplots(num_row, num_col, figsize=(10.8*num_col, 10.8*num_row))

    for i, image in enumerate(images):
        row_idx = i // 3
        col_idx = i % 3
        if num_row > 1:
            axes[row_idx, col_idx].imshow(image)
            axes[row_idx, col_idx].axis('off')
        else:
            axes[col_idx].imshow(image)
            axes[col_idx].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()