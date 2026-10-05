"""
Setup script for MSFP-Track (litepp package)

Multi-Scale Feature Pyramid with Adaptive Thresholds for Real-Time MOT
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README from root or litepp directory
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "MSFP-Track: Multi-Scale Feature Pyramid with Adaptive Thresholds for Real-Time MOT"

setup(
    name="litepp",
    version="1.0.0",
    author="AntVision AI Research",
    author_email="muhiddin@inha.ac.kr",
    description="MSFP-Track: Multi-Scale Feature Pyramid with Adaptive Thresholds for Real-Time MOT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://anonymous.4open.science/r/msfp-track",
    packages=find_packages(include=["litepp", "litepp.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "yolo": [
            "ultralytics>=8.0.0",
        ],
        "dev": [
            "pytest",
            "black",
            "isort",
            "flake8",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "scikit-learn>=1.0.0",
        ],
        "all": [
            "ultralytics>=8.0.0",
            "matplotlib>=3.5.0",
            "scikit-learn>=1.0.0",
        ],
    },
)
