python BrushNet/examples/brushnet/evaluate_brushnet.py \
    --base_model_path /home/work/llm/BrushNet/data/ckpt/realisticVisionV60B1_v51VAE \
    --brushnet_ckpt_path /home/work/llm/BrushNet/data/ckpt/segmentation_mask_brushnet_ckpt \
    --image_save_path runs/evaluation_result/BrushBench/brushnet_segmask/outside \
    --mapping_file /home/work/llm/BrushNet/data/BrushBench/mapping_file.json \
    --base_dir /home/work/llm/BrushNet/data/BrushBench \
    --mask_key outpainting_mask