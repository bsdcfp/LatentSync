python examples/evaluate_brushnet_sdxl.py \
    --base_model_path stabilityai/stable-diffusion-xl-base-1.0 \
    --brushnet_ckpt_path /home/work/llm/aigc_productbggen_v2/runs/logs/brushnetsdxl_segmentationmask/checkpoint-40000/brushnet \
    --image_save_path runs/evaluation_result/BrushBench/brushnet_sdxl_segmask_40k/outside \
    --mapping_file /home/work/llm/BrushNet/data/BrushBench/mapping_file.json \
    --base_dir /home/work/llm/BrushNet/data/BrushBench \
    --mask_key outpainting_mask