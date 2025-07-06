model='DiT-g/2'
params=" \
            --qk-norm \
            --model ${model} \
            --rope-img base512 \
            --rope-real \
            "

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 10.186.50.35  --master_port 23456 ./hydit/train_deepspeedcopy.py ${params}  "$@"