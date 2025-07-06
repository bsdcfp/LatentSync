pip install -i https://pypi.shopee.io/simple/ aip_model_repo

modelrepo login \
    --email xxx.xxx@shopee.com \
    --token xxxxxxxxx \
    --endpoint https://ais.mlp.shopee.io/api/modelRepo/v1/modelrepo/

modelrepo file uploadLFS \
 --model_id xxx \
 --project_id xxx \
 --version_name xxx \
 --version_desc xxx \
 -f ./models

mkdir models && cd models
modelrepo file download --model_id xxx --project_id xxx --version_id xxx --output_path .
cd ..
