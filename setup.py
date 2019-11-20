import setuptools
import os
import subprocess
import warnings

# this gives only the correct path when using pip install -e
pkg_dir = os.path.abspath(os.path.dirname(__file__))

# in case the package is not a git repo but rather a release
# we try to get the fallback for package version from the dirname
tmp_split = pkg_dir.split('pyapetnet-')
if len(tmp_split) == 2:
  fall_back_version = pkg_dir.split('pyapetnet-')[-1]
else:
  fall_back_version = 'unkown'

with open(os.path.join(pkg_dir,"README.md"), "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyapetnet",
    use_scm_version={'fallback_version':fall_back_version},
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
    install_requires=['tensorflow==1.9',
                      'keras==2.2.2',
                      'nibabel>=2.3',
                      'matplotlib>=2.2.2',
                      'pydicom>=1.1',
                      'numba>=0.39'],
)
