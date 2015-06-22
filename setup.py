from setuptools import setup
from setuptools import find_packages

DISTNAME = 'tyssue'
DESCRIPTION = 'tyssue is stronger than Arnold Schwarzenegge'
LONG_DESCRIPTION = 'tyssue is going to conquer the world soon '
MAINTAINER = 'Guillaume Gay'
MAINTAINER_EMAIL = 'gllm.gay@gmail.com'
URL = 'https://github.com/CellModels/tyssue'
LICENSE = 'BSD 3-Clause'
DOWNLOAD_URL = 'https://github.com/CellModels/tyssue.git'
VERSION = '0.1'
DEPENDENCIES = ["numpy >= 1.8",
                "scipy >= 0.15",
                ]

if __name__ == "__main__":

    setup(
        name=DISTNAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        url=URL,
        license=LICENSE,
        download_url=DOWNLOAD_URL,
        version=VERSION,

        classifiers=["Development Status :: 4 - Beta",
                     "Intended Audience :: Science/Research",
                     "License :: OSI Approved :: BSD License",
                     "Natural Language :: English",
                     "Operating System :: MacOS",
                     "Operating System :: Microsoft",
                     "Operating System :: POSIX :: Linux",
                     "Programming Language :: Python :: 3.4",
                     "Programming Language :: Python :: Implementation :: CPython",
                     "Topic :: Scientific/Engineering :: Artificial Intelligence",
                     "Topic :: Scientific/Engineering :: Bio-Informatics",
                     "Topic :: Scientific/Engineering :: Image Recognition",
                     "Topic :: Scientific/Engineering :: Medical Science Apps",
                     ],

        packages=find_packages('src'),
        package_dir={'': 'src'},
        zip_safe=False
    )
