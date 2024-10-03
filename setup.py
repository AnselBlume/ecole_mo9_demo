# setup.py
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="ecole_mo9_demo",
    version="0.1.0",
    author="Ansel Blume",
    author_email="blume5@illinois.edu",
    description="demo package for ecole_mo9_demo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"ecole_mo9_demo": "src"},
    packages=["ecole_mo9_demo"],
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
