torchrun --nproc_per_node 1 \
    --nnodes $WORLD_SIZE --node_rank $RANK \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    BrushNet/examples/brushnet/train_brushnet_sdxl.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
    --output_dir runs/train/brushnetsdxl_segmentationmask \
    --train_data_dir /home/work/datasets/BrushData \
    --resolution 512 \
    --learning_rate 5e-5 \
    --train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --mixed_precision fp16 \
    --tracker_project_name brushnet \
    --report_to tensorboard \
    --resume_from_checkpoint latest \
    --validation_steps 500 \
    --checkpointing_steps 500 \
    --checkpoints_total_limit 5