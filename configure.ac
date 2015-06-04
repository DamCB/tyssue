dnl Process this file with autoconf to produce a configure script.
dnl Adapted from https://git.skewed.de/count0/graph-tool/blob/master/configure.ac

dnl tyssue package version number
m4_define(tyssue_version_major, 0)
m4_define(tyssue_version_minor, 1)
m4_define(tyssue_version_micro, 1dev)

AC_INIT([tyssue],[tyssue_version_major().tyssue_version_minor().tyssue_version_micro()],[https://github.com/CellModels/tyssue/issues], [tyssue], [https://github.com/CellModels/tyssue])

TYSSUE_VERSION_MAJOR=tyssue_version_major()
TYSSUE_VERSION_MINOR=tyssue_version_minor()
TYSSUE_VERSION_MICRO=tyssue_version_micro()
AC_SUBST(TYSSUE_VERSION_MAJOR)
AC_SUBST(TYSSUE_VERSION_MINOR)
AC_SUBST(TYSSUE_VERSION_MICRO)

AC_CONFIG_SRCDIR(src)
AM_INIT_AUTOMAKE([foreign])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])
# AM_PROG_CC_C_O ## not needed since automake 1.14
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

AC_SEARCH_LIBS([strerror],[cposix])
AC_SUBST(CXXFLAGS,$CXXFLAGS)
AC_USE_SYSTEM_EXTENSIONS
AC_PROG_CXX
AC_LANG([C++])
AC_PROG_CC_STDC
AC_HEADER_STDC
AC_DISABLE_STATIC
LT_INIT
AC_PROG_INSTALL
AC_CANONICAL_TARGET

dnl turn on warnings
[CXXFLAGS="-Wall -Wextra -ftemplate-backtrace-limit=0 ${CXXFLAGS}"]

dnl Checks for options

dnl Turn debugging on or off
AC_MSG_CHECKING(whether to enable debug info...)

AC_ARG_ENABLE([debug], [AS_HELP_STRING([--enable-debug],[compile debug information [default=disabled] ])],
        if test $enableval = yes; then
           [AC_DEFINE([DEBUG], 1, [compile debug info])]
           [CXXFLAGS="${CXXFLAGS} -ggdb3"]
           [AC_MSG_RESULT(yes)]
        else
           [CXXFLAGS="${CXXFLAGS} -DNDEBUG"]
           [AC_MSG_RESULT(no)]
        fi
        ,
        [CXXFLAGS="${CXXFLAGS} -DNDEBUG"]
        [AC_MSG_RESULT(no)]
        )

dnl enable c++11
old_cxxflags="$CXXFLAGS"
[CXXFLAGS="${CXXFLAGS} -std=gnu++11"]
AC_MSG_CHECKING([whether C++ compiler supports -std=gnu++11])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([])],
    [AC_MSG_RESULT([yes])],
    [AC_MSG_RESULT([no])]
    [CXXFLAGS="${old_cxxflags} -std=gnu++0x"])

dnl set template depth
[CXXFLAGS="${CXXFLAGS} -ftemplate-depth-250"]

dnl disable deprecation warning, to silence some harmless BGL-related warnings
[CXXFLAGS="${CXXFLAGS} -Wno-deprecated -Wno-unknown-pragmas"]

AC_MSG_CHECKING(whether to enable function inlining...)

[NO_INLINE=no]
AC_ARG_ENABLE([inline], [AS_HELP_STRING([--disable-inline],[disable inline functions [default=enabled] ])],
        if test $enableval = no; then
           [AC_DEFINE([NO_INLINE], 1, [disable function inlining])]
           [CXXFLAGS="${CXXFLAGS} -fno-inline -fno-default-inline"]
           [NO_INLINE=yes]
           [AC_MSG_RESULT(no)]
        else
           [AC_MSG_RESULT(yes)]
        fi
        ,
        [AC_MSG_RESULT(yes)]
        )

AM_CONDITIONAL(NO_INLINE, test $NO_INLINE = yes)

AC_MSG_CHECKING(whether to enable optimizations...)

AC_ARG_ENABLE([optimization], [AS_HELP_STRING([--disable-optimization],[disable optimization [default=enabled] ])],
        if test $enableval = no; then
           [AC_MSG_RESULT(no)]
    else
           [CXXFLAGS="${CXXFLAGS} -O3"]
       [AC_MSG_RESULT(yes)]
    fi
        ,
        [CXXFLAGS="${CXXFLAGS} -O3"]
        [AC_MSG_RESULT(yes)]
        )

AC_MSG_CHECKING(whether to enable symbol visibility manipulation...)

AC_ARG_ENABLE([visibility], [AS_HELP_STRING([--disable-visibility],[disable symbol visibility manipulation [default=enabled] ])],
        if test $enableval = yes; then
           [CXXFLAGS="${CXXFLAGS} -fvisibility=default -fvisibility-inlines-hidden"]
       [AC_MSG_RESULT(yes)]
    else
           [AC_MSG_RESULT(no)]
    fi
        ,
        [CXXFLAGS="${CXXFLAGS} -fvisibility=default -fvisibility-inlines-hidden"]
        [AC_MSG_RESULT(yes)]
        )

[USING_OPENMP=yes]
[OPENMP_LDFLAGS=""]
AC_MSG_CHECKING(whether to enable parallel algorithms with openmp...)
AC_ARG_ENABLE([openmp], [AS_HELP_STRING([--enable-openmp],[enable openmp [default=disabled] ])],
        if test $enableval = yes; then
           [AC_MSG_RESULT(yes)]
           [AC_DEFINE([USING_OPENMP], 1, [using openmp])]
           [CXXFLAGS="${CXXFLAGS} -fopenmp"]
           [OPENMP_LDFLAGS=" -lgomp "]
        else
       [USING_OPENMP=no]
       [CXXFLAGS="${CXXFLAGS} -Wno-unknown-pragmas"]
           [AC_MSG_RESULT(no)]
        fi
        ,
        [CXXFLAGS="${CXXFLAGS} -Wno-unknown-pragmas"]
        [AC_MSG_RESULT(no)]
        )
AC_SUBST(OPENMP_LDFLAGS)

dnl Override prefix with default value if not set
if test "x${prefix}" = "xNONE"; then
  prefix=${ac_default_prefix}
fi

dnl Checks for programs.

dnl Checks for libraries.

dnl math library
AC_CHECK_LIB(m,main)

dnl bzip2 compression library
AC_CHECK_LIB(bz2,main)

dnl expat
AC_CHECK_LIB(expat,main)

dnl python
echo "================"
echo "Detecting python"
echo "================"
AM_PATH_PYTHON([3.4]) ## Hell with 2.7!!
AC_SUBST(PYTHON_VERSION)
PYTHON_FULL_VERSION=[`${PYTHON} -c "import sys; print(sys.version.split()[0])"`]
AC_SUBST(PYTHON_FULL_VERSION)
[TEMP_CPPFLAGS="${CPPFLAGS}"]
AC_PYTHON_DEVEL([== '${PYTHON_FULL_VERSION}'])
[CPPFLAGS="${CPPFLAGS} ${PYTHON_CPPFLAGS}"]
[LDFLAGS="${LDFLAGS} ${PYTHON_LDFLAGS}"]

# default directory for submodules
[MOD_DIR="${PYTHON_SITE_PKG}"]
AC_ARG_WITH([python-module-path], [AS_HELP_STRING([--with-python-module-path=DIR],
                    [directory where python module will be installed
                                [default=auto-detected] ])],
            MOD_DIR=$withval)
echo "tyssue will be installed at: ${MOD_DIR}"

[MOD_DIR="${MOD_DIR}/tyssue"]
AC_SUBST(MOD_DIR)

echo "==========================="
echo "Using python version: ${PYTHON_FULL_VERSION}"
echo "==========================="


dnl boost
AX_BOOST_BASE([[1.53.0]])
AX_BOOST_PYTHON
if test "$BOOST_PYTHON_LIB" = ""; then
   AC_MSG_ERROR([No usable boost::python found])
fi
AX_BOOST_IOSTREAMS
if test "$BOOST_IOSTREAMS_LIB" = ""; then
   AC_MSG_ERROR([No usable boost::iostreams found])
fi
AX_BOOST_REGEX
if test "$BOOST_REGEX_LIB" = ""; then
   AC_MSG_ERROR([No usable boost::regex found])
fi

AX_BOOST_GRAPH
if test "$BOOST_GRAPH_LIB" = ""; then
   AC_MSG_ERROR([No usable boost::graph found])
fi

[CPPFLAGS="${CPPFLAGS} ${BOOST_CPPFLAGS}"]

dnl CGAL
AC_CHECK_LIB(CGAL,main)
[CGAL_LIBADD="-lCGAL -lgmp"]
AC_SUBST(CGAL_LIBADD)
[CGAL_FLAGS="-frounding-math"]
AC_MSG_CHECKING(whether to cgal debbuging with valgrind...)
AC_ARG_ENABLE([cgal-valgrind], [AS_HELP_STRING([--enable-cgal-valgrind],[enable CGAL valgrind debugging [default=disabled] ])],
        if test $enableval = yes; then
           [AC_MSG_RESULT(yes)]
       [CGAL_FLAGS="${CGAL_FLAGS} -DCGAL_DISABLE_ROUNDING_MATH_CHECK=ON"]
        else
           [AC_MSG_RESULT(no)]
        fi
        ,
        [AC_MSG_RESULT(no)]
        )
AC_SUBST(CGAL_FLAGS)

dnl Checks for header files.

dnl numpy
AX_PYTHON_MODULE(numpy, fatal)
dnl [NUMPY_DIR="${PYTHON_SITE_PKG}/numpy"]
[NUMPY_DIR=`$PYTHON -c "import numpy; print(numpy.get_include())"`]
AC_ARG_WITH([numpy], [AS_HELP_STRING([--with-numpy=DIR],[directory where numpy is installed
                      [default=auto-detected] ])],
            NUMPY_DIR=$withval)
AC_CHECK_HEADER([${NUMPY_DIR}/numpy/arrayobject.h],
                [CPPFLAGS="${CPPFLAGS} -I${NUMPY_DIR}"],
                [AC_MSG_ERROR([Numpy extension header not found])],
                [#include "Python.h"])

dnl scipy
AX_PYTHON_MODULE(scipy, fatal)
dnl [SCIPY_DIR="${PYTHON_SITE_PKG}/scipy"]
[SCIPY_DIR=`$PYTHON -c "import os, scipy; print(os.path.dirname(scipy.__file__))"`]
AC_ARG_WITH([scipy], [AS_HELP_STRING([--with-scipy=DIR],[scipy install directory
                      [default=auto-detected] ])],
            [SCIPY_DIR=$withval])
AC_CHECK_HEADER([${SCIPY_DIR}/weave/scxx/object.h],
                [AC_DEFINE([HAVE_SCIPY], [1], [using scipy's weave])]
                [CPPFLAGS="${CPPFLAGS} -I${SCIPY_DIR}"],
                [AC_MSG_WARN([Scipy weave/scxx header not found! (Ignore this if python 3 is being used.)])])

dnl expat
AC_CHECK_HEADER([expat.h],
                [],
                [AC_MSG_ERROR([Expat header not found])])

dnl cgal
AC_CHECK_HEADER([CGAL/version.h],
                [],
                [AC_MSG_ERROR([CGAL headers not found])])

dnl removed check for cairo

dnl sparsehash
[SPARSEHASH_PREFIX=sparsehash]
AC_ARG_WITH([sparsehash-prefix], [AS_HELP_STRING([--with-sparsehash-prefix=prefix],
                    [include directory prefix for sparsehash
                                [default=sparsehash] ])],
            [SPARSEHASH_PREFIX=$withval])

[USING_SPARSEHASH=yes]
AC_MSG_CHECKING(whether to enable sparsehash...)
AC_ARG_ENABLE([sparsehash], [AS_HELP_STRING([--disable-sparsehash],[disable use of google's sparsehash [default=enabled] ])],
        if test $enableval = no; then
           [USING_SPARSEHASH=no]
           [AC_MSG_RESULT(no)]
        else
           [AC_MSG_RESULT(yes)]
        fi
        ,
        [AC_MSG_RESULT(yes)]
        )

if test "$USING_SPARSEHASH" = "yes"; then
   PKG_CHECK_MODULES(SPARSEHASH, [libsparsehash],
                     AC_DEFINE([HAVE_SPARSEHASH], [1], [Using google's sparsehash]),
                     [AC_MSG_RESULT([not found by pkgconfig, trying default...])])

   [SPARSEHASH_CFLAGS="${SPARSEHASH_CFLAGS} -DSPARSEHASH_PREFIX=${SPARSEHASH_PREFIX}"]
   AC_SUBST(SPARSEHASH_CFLAGS)
   AC_DEFINE_UNQUOTED([SPARSEHASH_INCLUDE(f)], [<${SPARSEHASH_PREFIX}/f>],
                      [Sparsehash include macro])
   [CXXFLAGS="${CXXFLAGS} ${SPARSEHASH_CFLAGS}"]
   [CPPFLAGS="${CPPFLAGS} ${SPARSEHASH_CFLAGS}"]
   AC_CHECK_HEADER([${SPARSEHASH_PREFIX}/dense_hash_set], [],
                   [AC_MSG_ERROR([sparsehash headers not found])])
fi

dnl Checks for typedefs, structures, and compiler characteristics.
dnl Checks for library functions.

dnl Checks for Additional stuff.

dnl Set PACKAGE SOURCE DIR in config.h.
packagesrcdir=`cd $srcdir && pwd`

dnl Set PACKAGE PREFIX
if test "x${prefix}" = "xNONE"; then
  packageprefix=${ac_default_prefix}
else
  packageprefix=${prefix}
fi

dnl Set PACKAGE DATA & DOC DIR
packagedatadir=share/${PACKAGE}
packagedocdir=doc/${PACKAGE}


dnl Subst PACKAGE_DATA_DIR.
NO_PREFIX_PACKAGE_DATA_DIR="${packagedatadir}"
AC_SUBST(NO_PREFIX_PACKAGE_DATA_DIR)
PACKAGE_DATA_DIR="${packageprefix}/${packagedatadir}"
AC_SUBST(PACKAGE_DATA_DIR)

dnl Subst PACKAGE_DOC_DIR.
NO_PREFIX_PACKAGE_DOC_DIR="${packagedocdir}"
AC_SUBST(NO_PREFIX_PACKAGE_DOC_DIR)
PACKAGE_DOC_DIR="${packageprefix}/${packagedocdir}"

if test "x${docdir}" = "xNONE"; then
  PACKAGE_DOC_DIR="${packageprefix}/${packagedocdir}"
else
  PACKAGE_DOC_DIR="${docdir}"
fi
AC_SUBST(PACKAGE_DOC_DIR)

dnl Subst PYTHON_DIR.
AC_DEFINE_UNQUOTED([INSTALL_PREFIX],"${prefix}", [python prefix])
AC_DEFINE_UNQUOTED([PYTHON_DIR], "${PYTHON_SITE_PKG}", [The directory name for the site-packages subdirectory of the standard Python install tree.])

AC_DEFINE_UNQUOTED([CXXFLAGS],"${CXXFLAGS}", [c++ compilation options])
AC_DEFINE_UNQUOTED([CPPFLAGS],"${CPPFLAGS}", [c++ preprocessor compilation options])
AC_DEFINE_UNQUOTED([LDFLAGS],"${LDFLAGS}", [linker options])

AC_DEFINE_UNQUOTED([PACKAGE_DATA_DIR], "${PACKAGE_DATA_DIR}", [package data dir])
AC_DEFINE_UNQUOTED([PACKAGE_DOC_DIR], "${PACKAGE_DOC_DIR}", [package doc dir])
AC_DEFINE_UNQUOTED([PACKAGE_SOURCE_DIR], "${packagesrcdir}", [package source dir])

AC_DEFINE_UNQUOTED([AUTHOR], "Guillaume Gay <guillaume@damcb.com>", [program author(s)])

AC_DEFINE_UNQUOTED([COPYRIGHT],
    "Copyright (C) 2015 Guillaume Gay\nThis is free software; see the source for copying conditions.  There is NO\nwarranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.",
        [copyright info])

# git information
AC_DEFINE([GIT_COMMIT], "esyscmd(git show | head -n 1 | sed 's/commit //' |  grep -o -e '.\{8\}' | head -n 1 |tr -d '\n')", [git HEAD commit hash])
AC_DEFINE([GIT_COMMIT_DATE], "esyscmd(git log -1 | head -n 3 | grep 'Date:' | sed s/'Date:   '// | tr -d '\n')", [git HEAD commit date])

# GCC version
AC_DEFINE([GCC_VERSION], [(__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)], [GCC version value])

# set some automake variables

# global CPP flags
AC_SUBST(CPPFLAGS)

# global CXX flags
AC_SUBST(CXXFLAGS)

# CPP flags for submodules
[MOD_CPPFLAGS="-I\$(srcdir) -I\$(srcdir)/.. \ ## -I\$(srcdir)/../../boost-workaround \
-DHAVE_CONFIG_H"]
AC_SUBST(MOD_CPPFLAGS)

# default LIBADD flags for submodules
[MOD_LIBADD="${PYTHON_LDFLAGS} -l${BOOST_IOSTREAMS_LIB} -l${BOOST_PYTHON_LIB} \
-l${BOOST_REGEX_LIB} ${OPENMP_LDFLAGS} -lexpat"]
AC_SUBST(MOD_LIBADD)

# needed for typeinfo objects to work across DSO boundaries.
# see http://gcc.gnu.org/faq.html#dso
[MOD_LDFLAGS="-module -avoid-version -export-dynamic -no-undefined -Wl,-E -Wl,--as-needed"]

# MacOS needs different options
if [[ `echo ${target_os} | grep -c -e "^darwin"` == 1 ]]; then
   [MOD_LDFLAGS="-module -dynamiclib"]
fi

AC_SUBST(MOD_LDFLAGS)


AC_CONFIG_FILES([
Makefile
src/Makefile
src/cpp-tyssue/Makefile
src/tyssue/Makefile
])
AC_OUTPUT

# ugly, dirty, smelly hack to force libtool to link to libstc++ before anything
# else, to avoid strange exception-related segfaults in
# __cxa_allocate_exception() see:
# http://www.nabble.com/preventing-boost::python::throw_error_already_set-td20958723.html ## 404
# http://wiki.fifengine.de/Segfault_in_cxa_allocate_exception ## 404

### the referenced bug seems closed for debian and ubuntu (from bug reports pointed here:
##  http://web.archive.org/web/20090304014049/http://wiki.fifengine.de/Segfault_in_cxa_allocate_exception)
dnl AC_MSG_NOTICE([Modifying libtool to correctly link libstdc++ before anything else...])
dnl eval "sed -i -e 's/-nostdlib/-nostdlib -lstdc++/g' libtool"
