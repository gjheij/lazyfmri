import os
from setuptools import setup, find_packages
PACKAGES = find_packages()

# Function to read the version from version.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "mypackage", "version.py")
    with open(version_file, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                # Extract the version string
                return line.split("=")[-1].strip().strip('"')


CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GPL-v3 License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering"
]

# Description should be a one-liner:
description = "LazyPlot: seaborn wrapper for plotting"
# Long description will go up on the pypi page
long_description = """

LazyPlot
========
linescanning is a package specifically built for the analysis of line-scanning
fMRI data. The idea is to acquire line data based on earlier acquired popu-
lation receptive field (pRF) data and minimal curvature in the brain, for
which we acquire functional runs with pRF-mapping and high resolution ana-
tomical scans that are processed with fmriprep, pRFpy, and FreeSurfer.

To get started using these components in your own software, please go to the
repository README_.

.. _README: https://github.com/tknapen/linescanning/master/README.md

License
=======
``linescanning`` is licensed under the terms of the GPL-v3 license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2019--, Tomas Knapen,
Spinoza Centre for Neuroimaging, Amsterdam.
"""

NAME = "lazyplot"
MAINTAINER = "Jurjen Heij"
MAINTAINER_EMAIL = "jurjenheij@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/gjheij/linescanning"
DOWNLOAD_URL = ""
LICENSE = "GPL3"
AUTHOR = "Jurjen Heij"
AUTHOR_EMAIL = "jurjenheij@gmail.com"
PLATFORMS = "OS Independent"
VERSION = get_version()
REQUIRES = [
    "matplotlib",
    "numpy",
    "typing",
    "seaborn",
    "nilearn",
    "lmfit",
    "scikit-learn",
    'nideconv @ git+https://github.com/gjheij/nideconv.git'
]

opts = dict(
    name=NAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    platforms=PLATFORMS,
    version=VERSION,
    packages=PACKAGES,
    install_requires=REQUIRES,
    requires=REQUIRES
)


if __name__ == '__main__':
    setup(**opts)
