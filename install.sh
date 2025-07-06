#!/bin/bash

# Install the main package in editable mode
echo "Installing shopee-aip-product-scene in editable mode..."
pip install -e .

# Install Shopee internal dependencies if needed
if [ "$1" = "--with-shopee-deps" ]; then
    echo "Installing Shopee internal dependencies..."
    python install_shopee_deps.py
fi

echo "Installation completed!" 