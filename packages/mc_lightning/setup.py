import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# do similar for requirements

setuptools.setup(
    name="mc_lightning",
    version="0.1",
    author="Jackson Nyman",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'sklearn',
	'pytorch-lightning',
    ],
)
