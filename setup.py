import setuptools
import os

# read content of README.md
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                       'README.md'),
          encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="pyapetnet",
    use_scm_version={'fallback_version': 'unkown'},
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
    author="Georg Schramm",
    author_email="georg.schramm@kuleuven.be",
    description=
    "a CNN for anatomy-guided deconvolution and denoising of PET images",
    long_description=long_description,
    license='MIT',
    long_description_content_type="text/markdown",
    url="https://github.com/gschramm/pyapetnet",
    packages=setuptools.find_packages(
        exclude=["demo_data", "figures", "pyapetnet_2d", "scripts_bow", "wip"
                 ]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        'SimpleITK>=2.0', 'pymirc>=0.22', 'tensorflow>=2.2',
        'nibabel>=3.0', 'matplotlib>=3.1', 'pydicom>=2.0'],
    entry_points={
        'console_scripts': [
            'pyapetnet_predict_from_nifti=pyapetnet.predict_from_nifti:main',
            'pyapetnet_predict_from_dicom=pyapetnet.predict_from_dicom:main',
            'pyapetnet_list_models=pyapetnet.list_models:main',
        ],
    },
    include_package_data=True,
)
