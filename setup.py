import os
import platform
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

DISTNAME = "tyssue"
DESCRIPTION = "tyssue is a living tissues, cell level, modeling library"
LONG_DESCRIPTION = (
    "tyssue uses the scientific python ecosystem to model"
    " epithelium at the cellular level"
)
MAINTAINER = "Guillaume Gay"
MAINTAINER_EMAIL = "guillaume@damcb.com"
URL = "https://github.com/DamCB/tyssue"
LICENSE = "GPL-3.0"
DOWNLOAD_URL = "https://github.com/DamCB/tyssue.git"

files = ["*.so*", "*.a*", "*.lib*", "config/*/*.json", "stores/*.*"]


# Version management copied form numpy
# Thanks to them!
MAJOR = 0
MINOR = 9
MICRO = 0
ISRELEASED = False
VERSION = f"{MAJOR}.{MINOR}.{MICRO}"


def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        git_revision = out.strip().decode("ascii")
    except OSError:
        git_revision = "Unknown"

    return git_revision


def get_version_info():
    """Adding the git rev number needs to be done inside write_version_py(),
    otherwise the import of tyssue.version messes up the build under Python 3.
    """
    fullversion = VERSION
    if os.path.exists(".git"):
        git_revision = git_version()
    elif os.path.exists("tyssue/version.py"):
        # must be a source distribution, use existing version file
        # read from it instead of importing to avoid importing
        # the whole package
        with open("tyssue/version.py", "r", encoding="utf-8") as version:
            for line in version.readlines():
                if line.startswith("git_revision"):
                    git_revision = line.split("=")[-1][2:-2]
                    break
            else:
                git_revision = "Unknown"
    else:
        git_revision = "Unknown"

    if not ISRELEASED:
        fullversion += ".dev0+" + git_revision[:7]

    return fullversion, git_revision


def write_version_py(filename="tyssue/version.py"):
    """Writes a version.py file with version info from setup.py and git"""
    fullversion, git_revision = get_version_info()

    with open(filename, "w", encoding="utf-8") as version:
        version.write(
            f"""
# THIS FILE IS GENERATED FROM tyssue SETUP.PY
#
short_version = '{VERSION}'
full_version = '{fullversion}'
git_revision = '{git_revision}'
release = {ISRELEASED}
if release:
    version = full_version
else:
    version = short_version
"""
        )


# Extension management from pybind/cmake_example
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        # Make build optionnal
        Extension.__init__(self, name, sources=[], optional=True)
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as exc:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            ) from exc

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j2"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )
        print(env["CXXFLAGS"], "\n")


ext_modules = [
    CMakeExtension("tyssue.collisions.cpp.c_collisions"),
    CMakeExtension("tyssue.generation.cpp.mesh_generation"),
]


write_version_py()
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
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=find_packages(),
    package_data={"tyssue": files},
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
