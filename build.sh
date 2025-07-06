#!/bin/bash

python setup.py bdist_wheel
python -m pyc_wheel ./dist/*.whl
pip install --force-reinstall ./dist/*.whl