FROM mambaorg/micromamba:latest
RUN micromamba install --yes --name base --channel conda-forge \
      pyapetnet~=1.5.1 && \
    micromamba clean --all --yes
