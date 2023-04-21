import os

from setuptools import setup

requires = open("requirements.txt", "r").readlines() if os.path.exists("requirements.txt") else open("./articulated_processing.egg-info/requires.txt", "r").readlines()
print("#-------------------    ", str(os.listdir("./")))
setup(
    name="articulated-processing",
    version="0.0.1",
    author="davidliyutong",
    author_email="davidliyutong@sjtu.edu.cn",
    description="Code for post processing data",
    packages=[
        "articulated_processing",
        "articulated_processing.client",
        "articulated_processing.cmd",
        "articulated_processing.datamodels",
        "articulated_processing.detector",
        "articulated_processing.ekf",
        "articulated_processing.utils",
    ],
    python_requires=">=3.7",
    install_requires=requires,
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown"
)