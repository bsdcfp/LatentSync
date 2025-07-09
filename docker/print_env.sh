#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Usage:
# "./print_env.sh" - prints to stdout
# "./print_env.sh > env.txt" - prints to file "env.txt"

print_key_info() {
echo "***Key System Information***"

# GPU Information
if command -v nvidia-smi &> /dev/null; then
    # Get GPU name and memory in MB, then convert to GB
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    GPU_MEMORY_GB=$((GPU_MEMORY_MB / 1024))
    echo "GPU: $GPU_NAME ${GPU_MEMORY_GB}GB"
else
    echo "GPU: Not available"
fi

# CPU Information  
if [ -f /proc/cpuinfo ]; then
    CPU_INFO=$(grep "model name" /proc/cpuinfo | head -1 | cut -d':' -f2 | sed 's/^ *//')
    echo "CPU: $CPU_INFO"
else
    echo "CPU: Not available"
fi

# OS Information
if [ -f /etc/os-release ]; then
    OS_INFO=$(grep "PRETTY_NAME" /etc/os-release | cut -d'"' -f2)
    echo "OS: $OS_INFO"
else
    echo "OS: $(uname -s) $(uname -r)"
fi

# CUDA Version - check multiple ways
CUDA_VERSION=""
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
elif [ -f /usr/local/cuda/version.txt ]; then
    CUDA_VERSION=$(cat /usr/local/cuda/version.txt | sed 's/CUDA Version \([0-9.]*\)/\1/')
elif [ -f /usr/local/cuda/version.json ]; then
    CUDA_VERSION=$(grep -o '"version"[[:space:]]*:[[:space:]]*"[^"]*"' /usr/local/cuda/version.json | sed 's/.*"\([0-9.]*\)".*/\1/' | head -1)
elif command -v nvidia-smi &> /dev/null; then
    # Try to get CUDA version from nvidia-smi
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9.]*\).*/\1/')
fi

if [ -n "$CUDA_VERSION" ]; then
    echo "CUDA: $CUDA_VERSION"
else
    echo "CUDA: Not available"
fi

# Python Version
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python -c "import sys; print('{0}.{1}.{2}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))")
    echo "Python: $PYTHON_VERSION"
else
    echo "Python: Not available"
fi

# Python packages - improved detection
if command -v python &> /dev/null; then
    # Get package versions
    packages=("torch" "diffusers" "transformers" "tokenizers" "accelerate" "flash_attn")
    for package in "${packages[@]}"; do
        VERSION=$(python -c "
import sys
try:
    import $package
    try:
        print($package.__version__)
    except AttributeError:
        # Some packages might not have __version__
        import pkg_resources
        print(pkg_resources.get_distribution('$package').version)
except ImportError:
    # Package not installed
    pass
except Exception:
    # Any other error
    pass
" 2>/dev/null)
        if [ -n "$VERSION" ]; then
            echo "$package: $VERSION"
        fi
    done
fi

echo
}

print_env() {
echo "**git***"
if [ "$(git rev-parse --is-inside-work-tree 2>/dev/null)" == "true" ]; then
git log --decorate -n 1
echo "**git submodules***"
git submodule status --recursive
else
echo "Not inside a git repository"
fi
echo

echo "***OS Information***"
cat /etc/*-release
uname -a
echo

echo "***GPU Information***"
nvidia-smi
echo

echo "***CPU***"
lscpu
echo

echo "***CMake***"
which cmake && cmake --version
echo

echo "***g++***"
which g++ && g++ --version
echo

echo "***nvcc***"
which nvcc && nvcc --version
echo

echo "***Python***"
which python && python -c "import sys; print('Python {0}.{1}.{2}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))"
echo

echo "***Environment Variables***"

printf '%-32s: %s\n' PATH $PATH

printf '%-32s: %s\n' LD_LIBRARY_PATH $LD_LIBRARY_PATH

printf '%-32s: %s\n' NUMBAPRO_NVVM $NUMBAPRO_NVVM

printf '%-32s: %s\n' NUMBAPRO_LIBDEVICE $NUMBAPRO_LIBDEVICE

printf '%-32s: %s\n' CONDA_PREFIX $CONDA_PREFIX

printf '%-32s: %s\n' PYTHON_PATH $PYTHON_PATH

echo


# Print conda packages if conda exists
if type "conda" &> /dev/null; then
echo '***conda packages***'
which conda && conda list
echo
# Print pip packages if pip exists
elif type "pip" &> /dev/null; then
echo "conda not found"
echo "***pip packages***"
which pip && pip list
echo
else
echo "conda not found"
echo "pip not found"
fi
}

# Print key information first
print_key_info

echo "<details><summary>Click here to see environment details</summary><pre>"
echo "     "
print_env | while read -r line; do
    echo "     $line"
done
echo "</pre></details>"
