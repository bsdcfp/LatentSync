from setuptools import find_packages, setup
import os

# 获取README内容
with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

# 建议维护latentsync/__version__.py，若无则手动指定
version = "1.6.0"  # TODO: 建议在latentsync/__version__.py维护版本号

# 读取requirements.txt
install_requires = []
if os.path.exists("requirements.txt"):
    with open("requirements.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("--extra-index-url"):
                install_requires.append(line)

    setup(
    name="latentsync",
        author="LatentSync Team",
        author_email="",
    description="End-to-end lip-sync method based on audio-conditioned latent diffusion models.",
        long_description=long_description,
        long_description_content_type="text/markdown",
    url="https://github.com/xdit-project/LatentSync",
        version=version,
    packages=find_packages(include=["latentsync*"]),
    install_requires=install_requires,
    include_package_data=True,
    python_requires=">=3.10",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
    )
