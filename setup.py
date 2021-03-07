import setuptools

setuptools.setup(
    name="pyapetnet",
    use_scm_version={'fallback_version':'unkown'},
    setup_requires=['setuptools_scm','setuptools_scm_git_archive'],
    author="Georg Schramm",
    author_email="georg.schramm@kuleuven.be",
    description="a CNN for anatomy-guided deconvolution and denoising of PET images",
    license='MIT',
    long_description_content_type="text/markdown",
    url="https://github.com/gschramm/pyapetnet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=['tensorflow>=2.2',
                      'nibabel>=3.0',
                      'matplotlib>=3.1',
                      'pydicom>=2.0',
                      'pymirc @ https://github.com/gschramm/pymirc/archive/v0.22.zip'],
    entry_points = {'console_scripts' : ['pyapetnet_predict_from_nifti=pyapetnet.command_line_tools:predict_from_nifti','pyapetnet_predict_from_dicom=pyapetnet.command_line_tools:predict_from_dicom', 'pyapetnet_list_models=pyapetnet.command_line_tools:list_models',],},
    include_package_data=True,
)
