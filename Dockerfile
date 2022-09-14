FROM python:3.10

# update pip3
RUN pip install --no-cache-dir --upgrade pip

# install pyapetnet from pypi
RUN pip install --no-cache-dir pyapetnet
