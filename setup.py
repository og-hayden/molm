"""
Setup configuration for the MOLM package.
"""
from setuptools import find_packages, setup

setup(
    name="molm",
    version="0.1.0",
    description="Modular Token Modification in Language Models",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "networkx>=3.1",
        "requests>=2.31.0",
        "nltk>=3.8.1",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "mypy>=1.4.1",
            "flake8>=6.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 