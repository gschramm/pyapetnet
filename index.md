## What is pyapetnet?

pyapetnet is a pure python package for training and use of convolutional
neural network that does anatomy-guided deconvolution and denoising of PET images
in image space.

The idea pyapetnet is to obtain the image quality of MAP PET reconstructions
using an anatomical prior (the asymmetric Bowsher prior) using a CNN in image space.
The latter has the advantage that (a) no access to PET raw data is needed and (b)
that the predictions are much faster compared to a classical iterative PET reconstruction.

The package contains already trained CNN models, the source code for predictions
from standard nifti and dicom images, and the source code to train your own model
and is published under [MIT license](https://github.com/gschramm/pyapetnet/blob/master/LICENSE).

## Live Demo

An interactive online demo that does not require any installation (just a google account)
can be run 
[here](https://colab.research.google.com/drive/17R84I3asw81FgbXUaqHMMkmA7HzwNvS2#scrollTo=crao9VE7Wiq3)

Note that this demo by default runs on purely simulated data for which the model was not trained.

## How to use it?

If you want to use the package yourself, get the latest release from our 
[github repository](https://github.com/gschramm/pyapetnet/releases) and follow the
installation instructions [here](https://github.com/gschramm/pyapetnet/blob/master/README.md).

Once installed, you can run a few demos that show how to make predictions from nifti or dicom
files, or how to train your own model.

## References

If you are using pyapetnet for your research, please cite our paper:

G. Schramm et al., "Approximating anatomically-guided PET reconstruction in image
space using a convolutional neural network", *under review* in NeuroImage
