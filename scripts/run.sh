python pipeline/run_sdxl_brushnet_ipadapter.py \
    --foreground examples/test_data/test_foreground.png \
    --template_image examples/test_data/test_template_image.png \
    --template_config examples/test_data/test_template_config.txt \
    --logdir runs/infe/apptool \
    --ckpt models/SDXL-BrushNet-checkpoint-10000
