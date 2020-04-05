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

Assuming that you have downloadad / received the latest
version of the pyapetnet package in a .zip archive pyapetnet-XX.zip
and saved in ```mydir```

```conda activate pyapetnet```

```cd mydir``` 

You should now be in the parent dir containing the pyapetnet
zip archive
 
For the CPU tensorflow version run:
```pip install pyapetnet-XX.zip/[tf]```

For the GPU tensorflow version run:
```pip install pyapetnet-XX.zip/[tf_gpu]```

which will install the pyapetnet package inside the virtual
conda environment.

To verify the installation you can execute

```conda activate pyapetnet```

```conda list```

and check whether pyapetnet is in the list of installed packages.
