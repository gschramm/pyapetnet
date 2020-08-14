# pyapetnet

A convolutional neurol network (CNN) to mimick the behavior of anatomy-guided PET 
reconstruction in image space.

## Authors

Georg Schramm
David Rigie

## License 

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Installation

We recommend to use the anaconda python distribution and to create a
conda virtual environment for pyapetnet.

The installation consists of three steps:
1. Installation of anaconda (miniconda) python distribution
2. Creation of the conda virtual environment with all dependencies
3. Installation of the pyapetnet package using pip

### Installation of anaconda (miniconda)

Download and install Miniconda from <https://docs.conda.io/en/latest/miniconda.html>.

Please use the ***Python 3.x*** installer and confirm that the installer
should run ```conda init``` at the end of the installtion process.

To test your miniconda installtion, open a new terminal and execute

```conda list```

which should list the installed basic python packages.

### Creation of the virtual conda environment

To create a virtual conda python=3.6 environment execute

```conda create -n pyapetnet python=3.6 ipython ```

To test the installation of the virual environment, execute
```conda activate pyapetnet```

### Installation of the pyapetnet package

Activate the virual conda environment
```conda activate pyapetnet```

To install the pyapetnet package run (replace X.XX with the latest release
version that can be found on https://github.com/gschramm/pyapetnet/releases)

```pip install https://github.com/gschramm/pyapetnet/archive/vX.XX.zip```

which will install the pyapetnet package inside the virtual
conda environment.

To test the installation run (inside python or ipython)

```python
import pyapetnet
print(pyapetnet.__version__)
print(pyapetnet.__file__) 
```

## Run demos

To test whether your installation works, you can run
the demo 000_predict_from_nifti.py demo (located in the
pyapetnet/demos subfolder). 
This demo shows hoe to do a prediction from nifti files. 
By default, it uses simulated data which is included in
the package.
