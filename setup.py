from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lcdp-sim",
    version="0.1.0",
    author="Yun Hong",
    author_email="hy20051123@gmail.com",
    description="Language-Conditioned Diffusion Policy for Robot Manipulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/16yunH/LCDP-Sim",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.23.0",
        "diffusers>=0.21.0",
        "transformers>=4.30.0",
        "opencv-python>=4.7.0",
        "zarr>=2.13.0",
        "h5py>=3.8.0",
        "imageio>=2.25.0",
        "matplotlib>=3.6.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "einops>=0.6.0",
        "hydra-core>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "sim": [
            "mani-skill2>=0.5.0",
            "gymnasium>=0.29.0",
            "gymnasium-robotics>=1.2.0",
        ],
    },
)
