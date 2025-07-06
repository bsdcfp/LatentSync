# Installation Guide

## Quick Start

### Basic Installation
```bash
# Install the main package in editable mode
pip install -e .

# Or use the install script
./install.sh
```

### Installation with Shopee Internal Dependencies
```bash
# Install with Shopee internal dependencies
./install.sh --with-shopee-deps

# Or manually install Shopee dependencies
python install_shopee_deps.py
```

## Development Installation

For development, you can install with additional development dependencies:

```bash
# Install with development tools
pip install -e ".[dev]"

# Install with all optional dependencies
pip install -e ".[all]"
```

## Dependencies

### Core Dependencies
The package uses flexible versioning to accommodate different environments:

- torch>=2.2.0
- accelerate>=0.30.0
- transformers>=4.40.0
- peft>=0.10.0
- PyWavelets>=1.4.0
- cuda-python>=12.0.0
- onnx>=1.15.0
- onnxruntime>=1.17.0
- polygraphy>=0.47.0
- onnxsim>=0.4.24
- onnx-graphsurgeon>=0.5.0
- pandas>=1.5.0
- pynvml>=11.0.0
- diffusers>=0.30.0
- numpy>=1.21.0
- pillow>=8.0.0
- safetensors>=0.3.0
- huggingface-hub>=0.20.0
- psutil>=5.8.0
- packaging>=20.0
- pyyaml>=6.0

### Optional Dependencies

#### Development Tools
- pytest>=6.0
- pytest-cov>=2.0
- black>=22.0
- isort>=5.0
- flake8>=3.8
- mypy>=0.900

#### Shopee Internal Dependencies
- aip-oneservice>=0.4.0
- mms-sdk>=1.0.0
- shopee-onemonitor>=0.1.0

#### TensorRT Support
- tensorrt>=8.0.0
- polygraphy>=0.47.0
- cuda-python>=12.0.0
- opencv-python>=4.5.0

## Version Strategy

### Flexible Versioning
The package uses `>=` version constraints instead of exact versions (`==`) to:
- Allow compatibility with different environments
- Reduce dependency conflicts
- Enable easier updates and maintenance
- Support both development and production environments

### Version Compatibility
The minimum versions are set based on:
- Current production environment compatibility
- Feature requirements
- Security considerations
- Performance optimizations

## Migration from setup.py

The old `setup.py` has been replaced with modern `pyproject.toml`. The old file is backed up as `setup.py.old`.

### Key Changes
1. **Modern Build System**: Uses `pyproject.toml` instead of `setup.py`
2. **Better Dependency Management**: Dependencies are clearly separated into core, dev, shopee, and tensorrt groups
3. **Flexible Versioning**: Uses `>=` instead of `==` for better compatibility
4. **Development Tools**: Integrated black, isort, mypy configuration
5. **Cleaner Installation**: Shopee dependencies are handled separately to avoid conflicts

### Backward Compatibility
If you need to use the old setup.py, you can:
```bash
# Restore the old setup.py
mv setup.py.old setup.py

# Install using the old method
python setup.py develop
```

## Troubleshooting

### Shopee Internal Dependencies
If you encounter issues with Shopee internal dependencies:

1. Ensure you have access to the Shopee internal PyPI
2. Check your network connection to `pypi.shopee.io`
3. Try installing dependencies manually:
   ```bash
   pip install aip-oneservice>=0.4.0 --index-url http://pypi.shopee.io/simple/ --trusted-host pypi.shopee.io
   ```

### Version Conflicts
If you encounter version conflicts:

1. Create a virtual environment
2. Install dependencies in the correct order
3. Use the `--no-deps` flag if needed:
   ```bash
   pip install -e . --no-deps
   pip install -r requirements.txt
   ```

### Environment Compatibility
To check your current environment compatibility:
```bash
# Check installed versions
pip list | grep -E "(torch|accelerate|transformers|diffusers)"

# Install with specific versions if needed
pip install -e . --constraint constraints.txt
``` 