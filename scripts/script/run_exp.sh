root_dir=$1
cd $root_dir

bash build_env/init_ais_run.sh $root_dir

mkdir -p /home/work/.cache/huggingface/hub
echo ${root_dir}/pretrain_model/models--stabilityai--stable-diffusion-xl-base-1.0/
ln -s ${root_dir}/pretrain_model/models--stabilityai--stable-diffusion-xl-base-1.0/ ~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0
echo ${root_dir}/pretrain_model/models--madebyollin--sdxl-vae-fp16-fix/
ln -s ${root_dir}/pretrain_model/models--madebyollin--sdxl-vae-fp16-fix/ ~/.cache/huggingface/hub/models--madebyollin--sdxl-vae-fp16-fix

bash script/train_sdxl_brushnet.sh