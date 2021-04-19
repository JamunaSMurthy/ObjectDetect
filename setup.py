"""
ObjectDetect Setup Configuration

A comprehensive package for object detection using Faster R-CNN and YOLO v1
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="objectdetect",
    version="0.1.0",
    author="Jamuna S Murthy",
    author_email="",
    description="Object Detection implementations: Faster R-CNN and YOLO v1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ObjectDetect",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/ObjectDetect/issues",
        "Documentation": "https://github.com/yourusername/ObjectDetect#readme",
        "Source Code": "https://github.com/yourusername/ObjectDetect",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: GPU",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8,<3.12",
    install_requires=[
        "tensorflow>=2.10.0",
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "opencv-python>=4.6.0",
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "tqdm>=4.60.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
            "matplotlib>=3.5.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
            "pylint>=2.15.0",
            "mypy>=0.990",
        ],
        "tf": [
            "tensorflow>=2.10.0",
            "tf-models-official>=2.10.0",
        ],
        "torch": [
            "torch>=1.12.0",
            "torchvision>=0.13.0",
        ],
    },
    keywords=[
        "object-detection",
        "faster-rcnn",
        "yolo",
        "computer-vision",
        "deep-learning",
        "machine-learning",
        "tensorflow",
        "pytorch",
        "neural-networks",
    ],
    include_package_data=True,
    zip_safe=False,
)
