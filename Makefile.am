## Process this file with automake to produce Makefile.in

SUBDIRS = src

tyssuedocdir = ${PACKAGE_DOC_DIR}
nobase_dist_tyssuedoc_DATA = \
	INSTALL.md \
	LICENCE.md \
	README.md \
	doc/bibliography/tyssue.bib

EXTRA_DIST = $(tyssuedoc_DATA) autogen.sh

# Copy all the spec files. Of cource, only one is actually used.
dist-hook:
	for specfile in *.spec; do \
		if test -f $$specfile; then \
			cp -p $$specfile $(distdir); \
		fi \
	done
