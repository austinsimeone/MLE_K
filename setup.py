from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="kohls exam",
    packages=find_packages(),
    description="exam for kohls mle presentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[],
    url="https://github.com/austin-simeone/MLE_K",
    python_requires=">=3.9",
)
