#!/usr/bin/env python3
"""
Install Shopee internal dependencies with custom index URL.
This script handles the installation of packages from Shopee's internal PyPI.
"""

import subprocess
import sys
import os

# Shopee internal PyPI configuration
INDEX_URL = "http://pypi.shopee.io/simple/"
TRUSTED_HOST = "pypi.shopee.io"

# Shopee internal dependencies (with flexible versioning)
SHOPEE_DEPS = [
    "aip-oneservice>=0.4.0",
    "mms-sdk>=1.0.0",
    "shopee-onemonitor>=0.1.0",
]

def install_with_custom_index(deps, index_url=INDEX_URL, trusted_host=TRUSTED_HOST):
    """Install packages with custom --index-url"""
    for dep in deps:
        try:
            print(f"Installing {dep} from {index_url}")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep,
                "--index-url", index_url,
                "--trusted-host", trusted_host
            ])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {dep}: {e}")
            return False
    return True

def main():
    """Main installation function"""
    print("Installing Shopee internal dependencies...")
    
    if install_with_custom_index(SHOPEE_DEPS):
        print("Successfully installed all Shopee dependencies!")
    else:
        print("Failed to install some Shopee dependencies.")
        sys.exit(1)

if __name__ == "__main__":
    main() 