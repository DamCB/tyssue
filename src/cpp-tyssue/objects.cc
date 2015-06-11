#include <CGAL/Linear_cell_complex.h>
#include <CGAL/Simple_cartesian.h>

#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>

#include "objects.hh"




struct World
{
  void set(std::string msg)
  {
    this->msg = msg;
  }
  std::string greet()
  {
    return msg;
  }
  std::string msg;
};




#include <boost/python.hpp>
using namespace boost::python;

void export_world()
{
  class_<World>("World")
    .def("greet", &World::greet)
    .def("set", &World::set);
}





// template <Epithelium>
Halfedge_handle add_triangle(Epithelium eptm)
{
  Halfedge_handle h;
  h = eptm.make_triangle();
  return h;
}



std::size_t get_num_vert(Epithelium &eptm)
{
  size_t Nv = eptm.size_of_vertices();
  return Nv;
}
// int main() {
//     Polyhedron P;
//     Halfedge_handle h = P.make_tetrahedron();

//     h->facet()->color = CGAL::RED;
//     return 0;
// }


void export_epithelium()
{

  //class_<Epithelium, bases<Poly> >("Epithelium")
  class_<Epithelium>("Epithelium")
    //.def("add_triangle", &Epithelium::make_triangle);
    .def("num_jvs", &Epithelium::size_of_vertices)
    .def("num_jes", &Epithelium::size_of_halfedges)
    .def("num_cells", &Epithelium::size_of_facets)
    ;
}
