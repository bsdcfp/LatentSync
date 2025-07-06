model='DiT-g/2'
params=" \
            --qk-norm \
            --model ${model} \
            --rope-img base512 \
            --rope-real \
            "
# accelerate launch \
#    --main_process_ip 10.243.126.15 \
#    --num_processes 1 \
#    --num_machines 1 \
#    --main_process_port 23456 \
#    --machine_rank 0 \
#    --use_deepspeed \
#    --mixed_precision bf16 \
#    --deepspeed_multinode_launcher standard \
#    hydit/train_deepspeed_controlnet3.py ${params}  "$@"

accelerate launch \
   --main_process_ip $MASTER_ADDR \
   --num_processes $WORLD_SIZE \
   --num_machines $WORLD_SIZE \
   --main_process_port $MASTER_PORT \
   --machine_rank $RANK \
   --use_deepspeed \
   --mixed_precision bf16 \
   --deepspeed_multinode_launcher standard \
   hydit/train_deepspeed_controlnet3.py ${params}  "$@"