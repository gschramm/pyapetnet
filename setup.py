import setuptools

setuptools.setup(
    name="pyapetnet",
    use_scm_version={'fallback_version':'unkown'},
    setup_requires=['setuptools_scm','setuptools_scm_git_archive'],
    author="Georg Schramm",
    author_email="georg.schramm@kuleuven.be",
    description="A CNN to mimick anatomy guided PET reconstruction in image space",
    license='MIT',
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
                      'nibabel==2.3',
                      'matplotlib==2.2.2',
                      'pydicom==1.1',
                      'numba==0.39'],
    extras_require={'tf': ['tensorflow==1.9'],
                    'tf_gpu': ['tensorflow-gpu==1.9'],
},
)
