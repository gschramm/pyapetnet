# latest tensorflow docker image, ships CPU version of tensorflow
FROM tensorflow/tensorflow:latest

# update pip3
RUN pip install --no-cache-dir --upgrade pip

# pyapetnet needs pymirc which needs numba which needs setuptools < 60
RUN pip install --no-cache-dir setuptools==59.8.0

## local install - not working yet
#COPY . /pyapetnet/
#RUN ls -la /pyapetnet
#RUN pip install /pyapetnet

# install pyapetnet from pypi
RUN pip install --no-cache-dir pyapetnet
