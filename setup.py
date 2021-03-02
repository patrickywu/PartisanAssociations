import setuptools
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="PartisanAssociations", # Replace with your own username
    version="0.0.1",
    author="Patrick Y. Wu, Walter R. Mebane, Jr., Logan Woods, Joseph Klaver, and Preston Due",
    author_email="pywu@umich.edu",
    description="Partisan associations using Twitter user bios",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/patrickywu/PartisanAssociations",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
