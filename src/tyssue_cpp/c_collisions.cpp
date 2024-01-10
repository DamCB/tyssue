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
    for (int i=0; i<(info_faces.shape[0]*info_faces.shape[1]); i=i+3)
    {
//        std::cout << i << std::endl;
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

void write_polygon_mesh(Mesh& mesh, std::string filename)
{
    CGAL::IO::write_polygon_mesh(filename, mesh, CGAL::parameters::stream_precision(17));
}


PYBIND11_MODULE(_collisions, m)
{

  m.def("write_polygon_mesh", &write_polygon_mesh);
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
                    .def("get_vertices",
                    [](Mesh& m)
                    {
                        std::vector<float> verts;
                        for (Mesh::Vertex_index vi : m.vertices()) {
                            K::Point_3 pt = m.point(vi);
                            verts.push_back((float)pt.x());
                            verts.push_back((float)pt.y());
                            verts.push_back((float)pt.z());
                        }
                        return verts;
                    })
                    .def("get_faces",
                    [](Mesh& m)
                    {
                        std::vector<uint32_t> indices;
                        for (Mesh::Face_index face_index : m.faces()) {
                            CGAL::Vertex_around_face_circulator<Mesh> vcirc(m.halfedge(face_index), m), done(vcirc);
                            do indices.push_back(*vcirc++); while (vcirc != done);
                        }
                        return indices;
                    })
    ;

}
