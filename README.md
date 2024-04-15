# pyapetnet

A convolutional neurol network (CNN) to mimick the behavior of anatomy-guided PET reconstruction in image space.

![architecture of pyapetnet](./figures/fig_1_apetnet.png)

## Authors

Georg Schramm, David Rigie

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Scientific Publication

Details about pyapetnet are published in [Schramm et al., "Approximating anatomically-guided PET reconstruction in image space using a convolutional neural network" ,NeuroImage Vol 224 2021](https://doi.org/10.1016/j.neuroimage.2020.117399).
If we you are using pyapetnet in scientific publications, we appreciate citation of this article.

## Installation

We recommend to install `pyapetnet` from `conda-forge` via

```
conda -c conda-forge install pyapetnet
```

Alternatively, `pyapetnet` can be also installed from `pypi` via

```
pip install pyapetnet
```

## Testing the installation

To test the installation activate your virtual environment and run

```python
import pyapetnet
print(pyapetnet.__file__)
```

If the installation was successful, a number of command line scripts all starting with pyapetnet\* to e.g. do predictions with the included trained models from nifti and dicom input images will be available.

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

The source code of the prediction scripts can be found in the `scripts` subfolder [here](src/pyapetnet/scripts).

## Training your own model

If you want to train your own model (from scratch or using transfer learning)
using your own data, have a look at our [training script](scripts/train_bow_cnn_tf2.py).
All input parameters (e.g. data sets to use) have to be specified in a config json file ([example here](scripts/train_cfg.json)). 
The input data sets have to be in nifti format and should be aligned already.
