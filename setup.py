import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyapetnet",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
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
    install_requires=['numpy>=1.15',
                      'nibabel>=2.3',
                      'scipy>=1.1',
                      'keras>=2.2.2',
                      'matplotlib>=2.2.2',
                      'pydicom>=1.1',
                      'h5py>=2.8',
                      'tensorflow>=1.9',
                      'numba>=0.39'],
)
