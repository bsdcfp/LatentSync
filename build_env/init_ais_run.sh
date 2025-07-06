root_dir=$1
cd $root_dir

# brushnet
cd BrushNet
pip install -e .
cd $root_dir

# pkgs
bash build_env/fix_cv2.sh
cd $root_dir

pip install -r build_env/requirements.txt
pip uninstall -y transofrmer-engine
