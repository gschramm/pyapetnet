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
We also provide a conda .yml file that allows to easily
create a conda environment containing all dependencies in the
tested versions.

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

```conda create -n py36-tf19-pyapetnet-cpu python=3.6```

To test the installation of the virual environment, execute

```conda activate py36-tf19-pyapetnet-cpu```

### Installation of the pyapetnet package

Assuming that you have downloaded / extracted the package
in ```mydir``` install the pyapetnet package via:

```conda activate py36-tf19-pyapetnet-cpu```

```cd mydir```

```pip install -e .```

which will install the pyapetnet package inside the virtual
conda environment.
Note that the ***-e*** is necesarry to pass the correct version
number to pip.

To verify the installation you can execute

```conda activate py36-tf19-pyapetnet-cpu```

```conda list```

and check whether pyapetnet is in the list of installed packages.
