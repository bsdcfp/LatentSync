#!/bin/bash

python setup.py bdist_wheel
python -m pyc_wheel ./dist/*.whl
twine upload --repository-url https://pypi.shopee.io -u shopee_aip_dpl@shopee.com -p shopeeaipsnndpl ./dist/*.whl

cd BrushNet
python setup.py bdist_wheel
python -m pyc_wheel ./dist/*.whl
twine upload --repository-url https://pypi.shopee.io -u shopee_aip_dpl@shopee.com -p shopeeaipsnndpl ./dist/*.whl
