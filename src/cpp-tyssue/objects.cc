#include <math.h>
#include <CGAL/Linear_cell_complex.h>
#include <CGAL/Simple_cartesian.h>

#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
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

template <class HDS>
class Build_hexagon : public CGAL::Modifier_base<HDS> {
public:
  Build_hexagon() {};
  void operator()( HDS& hds ){
    CGAL::Polyhedron_incremental_builder_3<HDS> B( hds, true);
    B.begin_surface(6, 1, 6);
    typedef typename HDS::Vertex   Vertex;
    typedef typename Vertex::Point Point;
    B.add_vertex( Point( 0, 1, 0) );
    B.add_vertex( Point( 0.5, sqrt(3)/2, 0) );
    B.add_vertex( Point( 0.5, -sqrt(3)/2, 0) );
    B.add_vertex( Point( 0, -1, 0) );
    B.add_vertex( Point( -0.5, -sqrt(3)/2, 0) );
    B.add_vertex( Point( 0.5, -sqrt(3)/2, 0) );
    B.begin_facet();
    B.add_vertex_to_facet( 0);
    B.add_vertex_to_facet( 1);
    B.add_vertex_to_facet( 2);
    B.add_vertex_to_facet( 3);
    B.add_vertex_to_facet( 4);
    B.add_vertex_to_facet( 5);
    B.end_facet();
    B.end_surface();
  };
};


void make_hexagon(Epithelium &eptm){
  Build_hexagon<HalfedgeDS> hexagon;
  eptm.delegate( hexagon);
};



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
