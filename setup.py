#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="GCDM-SBDD",
    version="0.0.1",
    description="A geometry-complete diffusion model for structure-based drug design.",
    author="Alex Morehead",
    author_email="acmwhb@umsystem.edu",
    url="https://github.com/BioinfoMachineLearning/GCDM-SBDD",
    install_requires=["pytorch-lightning"],
    packages=find_packages(),
)