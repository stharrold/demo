# Build on public images. Order by size of layer (most required IO to least required IO)
# Follow https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices/
#
# TODO:
# * Minimize image size with miniconda (https://github.com/stharrold/demo/issues/5)

# Continuum IO's Anaconda Python 3:
# https://hub.docker.com/r/continuumio/anaconda3/
FROM continuumio/anaconda3:4.1.1

# Update the Linux kernel and install other packages.
RUN apt-get update -y && \
    apt-get install -y \
        build-essential \
        less \
        locales \
        lsof && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install extra Python packages with "conda" package manager if possible;
# otherwise install with Python Package Index 'pip' package manager.
# * Use 'pip' for 'seaborn' to get latest version
#   (http://seaborn.pydata.org/installing.html).
# * Upgrade beautifulsoup4 and html5lib through pip due to recent deprecations.
#   http://stackoverflow.com/questions/38447738/beautifulsoup-html5lib-module-object-has-no-attribute-base
RUN conda install -y \
        jupyter \
        graphviz && \
    conda install -y -c r \
        r-essentials \
        r-e1071 \
        r-rocr
RUN pip install --upgrade pip && \
    pip install \
        astroML \
        astroML_addons \
        fastcluster \
        geopy \
        seaborn && \
    pip install --upgrade \
        beautifulsoup4

# Copy codebase into image.
# Start a bash shell from /opt/demo if container is initialized without a command.
COPY . /opt/demo/.
WORKDIR /opt/demo
CMD ["bash"]
