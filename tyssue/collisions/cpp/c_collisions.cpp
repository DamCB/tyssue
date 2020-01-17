#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <boost/lexical_cast.hpp>

#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <iostream>


#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>

namespace py = pybind11;
namespace PMP = CGAL::Polygon_mesh_processing;

using K                     = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3               = K::Point_3;
using Mesh                  = CGAL::Surface_mesh<Point_3>;
using face_descriptor       = boost::graph_traits<Mesh>::face_descriptor;

typedef Mesh::Vertex_index vertex_descriptor;
typedef Mesh::Face_index face_descriptor;


Mesh sheet_to_surface_mesh(py::array_t<double> vertices, py::array_t<double> faces)
{

    Mesh mesh;

    // scan vertices numpy array
    py::buffer_info info_vertices = vertices.request();
    for (int i=0; i<info_vertices.shape[0]*3; i=i+3)
    {
        mesh.add_vertex(Point_3(((double*)info_vertices.ptr)[i],
                                ((double*)info_vertices.ptr)[i+1],
                                ((double*)info_vertices.ptr)[i+2]));
    }

    // scan faces numpy array
    py::buffer_info info_faces = faces.request();
    for (int i=0; i<info_faces.shape[0]; i=i+3)
    {

      vertex_descriptor u = vertex_descriptor(((double*)info_faces.ptr)[i]);
      vertex_descriptor v = vertex_descriptor(((double*)info_faces.ptr)[i+1]);
      vertex_descriptor w = vertex_descriptor(((double*)info_faces.ptr)[i+2]);
      face_descriptor f = mesh.add_face(u,v,w);

    }

    return mesh;
}


bool does_self_intersect (Mesh& mesh)
{
    return PMP::does_self_intersect(mesh, PMP::parameters::vertex_point_map(get(CGAL::vertex_point, mesh)));
}

std::vector<std::tuple<int, int>> self_intersections(Mesh& mesh)

{
    std::vector<std::pair<face_descriptor, face_descriptor>> intersected_tris;
    PMP::self_intersections(mesh, std::back_inserter(intersected_tris));

    std::vector<std::tuple<int, int>> list;
    for (std::size_t i=0; i< intersected_tris.size(); i++)
    {
        std::tuple<int, int> tmp = std::make_tuple(intersected_tris[i].first, intersected_tris[i].second);
        list.push_back(tmp);
    }
    return list;

}


PYBIND11_MODULE(c_collisions, m)
{
    m.def("sheet_to_surface_mesh", &sheet_to_surface_mesh);

    m.def("does_self_intersect", &does_self_intersect);

    m.def("self_intersections", &self_intersections);

    py::class_<Mesh>(m, "Mesh")
                .def(py::init<>())
                .def(py::init<Mesh&>())
                .def("number_of_vertices",
                     [](Mesh& m)
                     {
                        return m.number_of_vertices();
                     })
                .def("number_of_faces",
                     [](Mesh& m)
                     {
                        return m.number_of_faces();
                     })
    ;

}
