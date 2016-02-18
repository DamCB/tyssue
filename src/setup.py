from setuptools import setup
from setuptools import find_packages

DISTNAME = 'tyssue'
DESCRIPTION = 'tyssue is a living tissues, cell level, modeling library'
LONG_DESCRIPTION = ('tyssue uses the scientific python ecosystem and CGAL'
                    ' LinearCellComplex library to model epithelium at the'
                    ' cellular level')
MAINTAINER = 'Guillaume Gay'
MAINTAINER_EMAIL = 'guillaume@damcb.com'
URL = 'https://github.com/CellModels/tyssue'
LICENSE = 'MPL'
DOWNLOAD_URL = 'https://github.com/CellModels/tyssue.git'
# VERSION = '${Tyssue_VERSION}'
VERSION = '0.1'


files = ['*.so*', '*.a*', '*.lib*',
         'config/*/*.json','stores/*.*']


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
                     "License :: OSI Approved :: MPL v2.0",
                     "Natural Language :: English",
                     "Operating System :: MacOS",
                     "Operating System :: Microsoft",
                     "Operating System :: POSIX :: Linux",
                     "Programming Language :: Python :: 3.4",
                     "Programming Language :: Python :: Implementation :: CPython",
                     "Topic :: Scientific/Engineering :: Artificial Intelligence",
                     "Topic :: Scientific/Engineering :: Bio-Informatics",
                     "Topic :: Scientific/Engineering :: Medical Science Apps",
                     ],

        packages=find_packages(),
        package_data={'tyssue': files},
        include_package_data=True,
        zip_safe=False
    )
