# Workflow to build the parallelproj C/CUDA libs (incl. installation of CUDA)
name: python build

on:
  push:
    branches: [ "master" ]
  pull_request:
    branches: [ "master" ]

jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11']

    runs-on: ${{ matrix.os }}
    env:
      MPLBACKEND: Agg  # https://github.com/orgs/community/discussions/26434

    steps:
    - uses: actions/checkout@v3

    - name: Set up python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install package
      run: |
        python -m pip install --upgrade pip
        pip install .
      
    - name: Run nifti prediction
      run: |
        cd demo_data
        pyapetnet_predict_from_nifti brainweb_06_osem_cropped.nii brainweb_06_t1_cropped.nii S2_osem_b10_fdg_pe2i --no_coreg
