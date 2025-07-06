# AIGC_ProductBgGen_v2

## 1. Init ENV

AIS Image: 
```
harbor.shopeemobile.com/aip/aip-image-hub/aip-prod/projects/0/pytorch2.0-cu12.1-py3.10-trt8.6:py3.10-cu12.1-pt2.0-trt8.6-vscode1.82.2-d540733753
```

Run init script:
```
bash build_env/init_ais_run.sh {code_root_dir}
```

## 2. Train

Run AIS experiment:
```
bash script/run_exp.sh {code_root_dir}
```

## 3. Inference

```
python pipeline/run_sdxl_brushnet_ipadapter.py --foreground examples/test_data/test_foreground.png --template_image examples/test_data/test_template_image.png --template_config examples/test_data/test_template_config.txt --logdir runs/infe/apptool --ckpt {ckpt_path}
```