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


#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/Complex_2_in_triangulation_3.h>
#include <CGAL/make_surface_mesh.h>
#include <CGAL/Implicit_surface_3.h>

// default triangulation for Surface_mesher
typedef CGAL::Surface_mesh_default_triangulation_3 Tr;
// c2t3
typedef CGAL::Complex_2_in_triangulation_3<Tr> C2t3;
typedef Tr::Geom_traits GT;
typedef GT::Sphere_3 Sphere_3;
typedef GT::Point_3 Point_3;
typedef GT::FT FT;
typedef FT (*Function)(Point_3);
typedef CGAL::Implicit_surface_3<GT, Function> Surface_3;
//typedef C2t3::Vertex_handle vertex_descriptor;
//using vertex_descriptor       = boost::graph_traits<C2t3>::vertex_descriptor;


FT sphere_function (Point_3 p) {
  const FT x2=p.x()*p.x(), y2=p.y()*p.y(), z2=p.z()*p.z();
  return x2+y2+z2-1;
}

namespace py = pybind11;

std::vector<std::tuple<double, double, double>> make_spherical(double num_points) {
  Tr tr;            // 3D-Delaunay triangulation
  C2t3 c2t3 (tr);   // 2D-complex in 3D-Delaunay triangulation
  // defining the surface
  Surface_3 surface(sphere_function,             // pointer to function
                    Sphere_3(CGAL::ORIGIN, 2.)); // bounding sphere
  // Note that "2." above is the *squared* radius of the bounding sphere!
  // defining meshing criteria
  double r_max = std::sqrt(1 / (num_points * 0.12));
  CGAL::Surface_mesh_default_criteria_3<Tr> criteria(30, r_max, r_max);
  // meshing surface
  CGAL::make_surface_mesh(c2t3, surface, criteria, CGAL::Non_manifold_tag());
  std::vector<std::tuple<double, double, double>> points;
  Tr::Finite_vertices_iterator it;
  for (it = tr.finite_vertices_begin(); it != tr.finite_vertices_end(); it++)
    {
      Point_3 p = tr.point(it);
      std::tuple<double, double, double> tmp = std::make_tuple(p.x(), p.y(), p.z());
      points.push_back(tmp);
    }
  return points;
}


PYBIND11_MODULE(mesh_generation, m)
{
    m.def("make_spherical", &make_spherical);
}
