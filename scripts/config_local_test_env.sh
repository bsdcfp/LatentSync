pip install nvidia-pyindex

mkdir output

python3 -m pip install --no-cache-dir -U pip

pip install --no-cache-dir -r requirements.txt

pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install -e git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata

pip install --no-cache-dir --trusted-host pypi.shopee.io -i http://pypi.shopee.io/simple/ aip-oneservice==0.4.6
pip install --no-cache-dir --trusted-host pypi.shopee.io -i http://pypi.shopee.io/simple/ mms-sdk==1.0.30

# pip install --no-cache-dir nvidia-pyindex
# python3 -m pip install --pre --upgrade --extra-index-url https://pypi.nvidia.com tensorrt==9.3.0.post12.dev1

python3 -m pip install colored
pip3 install nvtx

apt-get update && apt-get install -y libgl1

bash install_tensorrt.sh
