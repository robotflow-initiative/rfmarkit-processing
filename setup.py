import os

from setuptools import setup

requires = open("requirements.txt", "r").readlines() if os.path.exists("requirements.txt") else open("./markit_processing.egg-info/requires.txt", "r").readlines()
print("#-------------------    ", str(os.listdir("./")))
setup(
    name="markit-processing",
    version="0.0.1",
    author="davidliyutong",
    author_email="davidliyutong@sjtu.edu.cn",
    description="Code for post processing data",
    packages=[
        "markit_processing",
        "markit_processing.client",
        "markit_processing.cmd",
        "markit_processing.datamodels",
        "markit_processing.detector",
        "markit_processing.ekf",
        "markit_processing.utils",
    ],
    python_requires=">=3.7",
    install_requires=requires,
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown"
)