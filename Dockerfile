# latest tensorflow docker image, ships CPU version of tensorflow
FROM tensorflow/tensorflow:latest

# update pip3
RUN pip install --no-cache-dir --upgrade pip

# install pyapetnet from pypi
RUN pip install --no-cache-dir pyapetnet
