model='DiT-g/2'
params=" \
            --qk-norm \
            --model ${model} \
            --rope-img base512 \
            --rope-real \
            "
# deepspeed --hostfile /home/work/donnhe2/HunyuanDiT/hydit/hostfile hydit/train_deepspeed.py ${params}  "$@"
# deepspeed --include="localhost:7" hydit/train_deepspeed.py ${params}  "$@"
# deepspeed --hostfile /home/work/donnhe2/HunyuanDiT/hydit/hostfile hydit/train_deepspeed.py ${params}  "$@"
# deepspeed --hostfile /home/work/donnhe2/HunyuanDiT/hydit/hostfile hydit/train_deepspeed.py ${params}  "$@"
accelerate launch \
   --main_process_ip 10.243.126.62 \
   --num_processes 5 \
   --num_machines 5 \
   --main_process_port 23458 \
   --machine_rank 4 \
   --use_deepspeed \
   --mixed_precision bf16 \
   --deepspeed_multinode_launcher standard \
   hydit/train_deepspeed_lcm.py ${params}  "$@"

# torchrun --nproc_per_node 1 \
#          --nnodes 2 \
#          --node_rank $RANK \
#          --master_addr $MASTER_ADDR \
#          --master_port $MASTER_PORT \
#          hydit/train_deepspeed.py ${params}  "$@"


# torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 10.186.53.69  --master_port 23456 ./hydit/train_deepspeedcopy.py ${params}  "$@"