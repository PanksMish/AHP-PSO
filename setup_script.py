"""
Setup script for AHP-PSO Object Detection
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ahp-pso-detection",
    version="1.0.0",
    author="Pankaj Mishra, V Venkataramanan, Anand Nayyar",
    author_email="pankaj.mishra@somaiya.edu",
    description="Adaptive Hybrid Particle Swarm Optimization for Real-Time Object Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ahp-pso-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ahp-pso-detect=main:main",
            "ahp-pso-benchmark=experiments.run_all_algorithms:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
    },
)
