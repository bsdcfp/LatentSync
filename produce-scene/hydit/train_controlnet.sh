task_flag="canny_controlnet"                                   # the task flag is used to identify folders.
control_type=canny
resume_module_root=/home/work/donnhe2/HunyuanDiT/log_EXP/036-dit_g2_full_1024p/pytorch_model_50pre_768_lcm.bin  # checkpoint root for resume
index_file=/path/to/your/indexfile                             # index file for dataloader
results_dir=./log_EXP                                          # save root for results
batch_size=8                                                   # training batch size
image_size=768                                                # training image resolution
grad_accu_steps=1                                              # gradient accumulation
warmup_num_steps=0                                             # warm-up steps
lr=1e-5                                                      # learning rate
ckpt_every=2500                                               # create a ckpt every a few steps.
ckpt_latest_every=5000                                         # create a ckpt named `latest.pt` every a few steps.
epochs=3                                                     # total training epochs


sh $(dirname "$0")/run_g_controlnet.sh \
    --task-flag ${task_flag} \
    --control-type ${control_type} \
    --noise-schedule scaled_linear --beta-start 0.00085 --beta-end 0.018 \
    --predict-type v_prediction \
    --uncond-p 0.44 \
    --uncond-p-t5 0.44 \
    --index-file ${index_file} \
    --lr ${lr} \
    --batch-size ${batch_size} \
    --image-size ${image_size} \
    --global-seed 999 \
    --grad-accu-steps ${grad_accu_steps} \
    --warmup-num-steps ${warmup_num_steps} \
    --use-flash-attn \
    --use-fp16 \
    --results-dir ${results_dir} \
    --resume \
    --resume-module-root ${resume_module_root} \
    --epochs ${epochs} \
    --ckpt-every ${ckpt_every} \
    --ckpt-latest-every ${ckpt_latest_every} \
    --log-every 50 \
    --deepspeed \
    --deepspeed-optimizer \
    --use-zero-stage 2 \
    --gradient-checkpointing \
    "$@"
