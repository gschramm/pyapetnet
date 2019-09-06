import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyapetnet",
    version="0.1",
    author="Georg Schramm",
    author_email="georg.schramm@kuleuven.be",
    description="A CNN to mimick anatomy guided PET reconstruction in image space",
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gschramm/pyapetnet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy',
                      'nibabel',
                      'scipy',
                      'keras',
                      'matplotlib',
                      'pydicom',
                      'h5py',
                      'numba'],
)
