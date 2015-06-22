// tyssue
//
// adapted from graph_tool Copyright (C) 2006-2015 Tiago de Paula Peixoto <tiago@skewed.de>
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#define NUMPY_EXPORT
#include "tyssue/numpy_bind.hh"
#include "tyssue/objects.hh"

#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "bytesobject.h"

using namespace std;
using namespace boost;
using namespace boost::python;

// struct LibInfo
// {
//     string GetName()      const {return PACKAGE_NAME;}
//     string GetAuthor()    const {return AUTHOR;}
//     string GetCopyright() const {return COPYRIGHT;}
//     string GetVersion()   const {return VERSION " (commit " GIT_COMMIT
//                                         ", " GIT_COMMIT_DATE ")";}
//     string GetLicense()   const {return "GPL version 3 or above";}
//     string GetCXXFLAGS()  const {return CPPFLAGS " " CXXFLAGS " " LDFLAGS;}
//     string GetInstallPrefix() const {return INSTALL_PREFIX;}
//     string GetPythonDir() const {return PYTHON_DIR;}
//     string GetGCCVersion() const
//     {
//         stringstream s;
//         s << __GNUC__ << "." << __GNUC_MINOR__ << "." <<  __GNUC_PATCHLEVEL__;
//         return s.str();
//     }
// };



// numpy array interface weirdness
void* do_import_array()
{
    import_array1(NULL);
    return NULL;
}


void export_world();

void export_epithelium();


BOOST_PYTHON_MODULE(libtyssue_core)
{
    using namespace boost::python;

    // numpy
    do_import_array();
    export_world();
    export_epithelium();
    // class_<LibInfo>("mod_info")
    //     .add_property("name", &LibInfo::GetName)
    //     .add_property("author", &LibInfo::GetAuthor)
    //     .add_property("copyright", &LibInfo::GetCopyright)
    //     .add_property("version", &LibInfo::GetVersion)
    //     .add_property("license", &LibInfo::GetLicense)
    //     .add_property("cxxflags", &LibInfo::GetCXXFLAGS)
    //     .add_property("install_prefix", &LibInfo::GetInstallPrefix)
    //     .add_property("python_dir", &LibInfo::GetPythonDir)
    //     .add_property("gcc_version", &LibInfo::GetGCCVersion);
}
