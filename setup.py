from setuptools import setup, find_packages
import sys, os

setup(
    name="forge",
    description="Machine Learning Experiment Management",
    version="0.1",
    author="Adam Kosiorek",
    license="GLPv3",
    python_requires=">=3.6",
    install_requires=[
        "dm_sonnet",
        "tensorflow",
        "torch",
        "torchvision",
        "numpy",
        "attrdict",
        "simplejson",
        "future",
    ],
    packages=find_packages(),
    long_description=open("README.md").read(),
)
