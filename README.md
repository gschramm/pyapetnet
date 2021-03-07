# pyapetnet

A convolutional neurol network (CNN) to mimick the behavior of anatomy-guided PET reconstruction in image space.

![](./figures/fig_1_apetnet.png | width=800)

## Authors

Georg Schramm, David Rigie

## License 

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Scientific Publication

Details about pyapetnet are published in [Schramm et al., "Approximating anatomically-guided PET reconstruction in image space using a convolutional neural network" ,NeuroImage Vol 224 2021](https://doi.org/10.1016/j.neuroimage.2020.117399).
If we you are using pyapetnet in scientific publications, we appreciate citation of this article.

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
```
conda list
```
which should list the installed basic python packages.

### Creation of the virtual conda environment

To create a virtual conda python=3.8 environment execute
```
conda create -n pyapetnet python=3.8 ipython
```
You can also you a newer version of python, if supported
by tensorflow.
To test the installation of the virual environment, execute
```
conda activate pyapetnet
```

### Installation of the pyapetnet package

Activate the virual conda environment
```
conda activate pyapetnet
```
To install the pyapetnet package run (replace X.XX with the latest release
version that can be found on https://github.com/gschramm/pyapetnet/releases)
```
pip install https://github.com/gschramm/pyapetnet/archive/vX.XX.zip
```
which will install the pyapetnet package inside the virtual
conda environment.

To test the installation run (inside python or ipython)
```python
import pyapetnet
print(pyapetnet.__version__)
print(pyapetnet.__file__) 
```

If the installation was successful, a number of command line scripts all starting with pyapetnet* to e.g. do predictions with the included trained models from nifti and dicom input images will be available.

## Getting started - running your first prediction with pre-trained models

To run a prediction using one of included pre-trained networks and **nifti images**, run e.g.:
```
pyapetnet_predict_from_nifti osem.nii t1.nii S2_osem_b10_fdg_pe2i --show
```
Use the following to get information on the (optional) input arguments
```
pyapetnet_predict_from_nifti -h
```
To get a list of available pre-trained models run
```
pyapetnet_list_models
```

To make predictions from **dicom images**, use
```
pyapetnet_predict_from_dicom osem_dcm_dir t1_dcm_dir S2_osem_b10_fdg_pe2i --show
```
The source code of the prediction scripts can be found in the *command_line_tools* sub module.
