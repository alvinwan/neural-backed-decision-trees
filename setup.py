import setuptools

VERSION = "0.0.4"

with open("requirements.txt", "r") as f:
    install_requires = f.readlines()


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="nbdt",
    version=VERSION,
    author="Alvin Wan",  # TODO: proper way to list all paper authors?
    author_email="hi@alvinwan.com",
    description="Making decision trees competitive with state-of-the-art "
    "neural networks on CIFAR10, CIFAR100, TinyImagenet200, "
    "Imagenet. Transform any image classification neural network "
    "into an interpretable neural-backed decision tree.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alvinwan/neural-backed-decision-trees",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    download_url="https://github.com/alvinwan/neural-backed-decision-trees/archive/%s.zip"
    % VERSION,
    scripts=["nbdt/bin/nbdt-hierarchy", "nbdt/bin/nbdt-wnids", "nbdt/bin/nbdt"],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
    include_package_data=True,
)
