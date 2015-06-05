## Process this file with automake to produce Makefile.in

#SUBDIRS = centrality clustering community

AM_CPPFLAGS =\
    -I$(srcdir)/.. \
    -DHAVE_CONFIG_H

AM_CXXFLAGS =\
    -Wall \
    $(PYTHON_CPPFLAGS) \
    $(BOOST_CPPFLAGS)

AM_CFLAGS=$(AM_CXXFLAGS)

libtyssue_coredir = $(MOD_DIR)
libtyssue_core_LTLIBRARIES = libtyssue_core.la

libtyssue_core_la_SOURCES = \
    objects.cc \
    tyssue_bind.cc

libtyssue_core_la_includedir = $(MOD_DIR)/include
libtyssue_core_la_include_HEADERS = \
    ../../config.h

libtyssue_core_la_LIBADD = $(MOD_LIBADD)
libtyssue_core_la_LDFLAGS = $(MOD_LDFLAGS)
