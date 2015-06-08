
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>



template <class Refs>
struct Juction_vertex : public CGAL::HalfedgeDS_vertex_base<Refs> {
  bool is_active;
};

template <class Refs>
struct Junction_edge : public CGAL::HalfedgeDS_halfedge_base<Refs> {
  double line_tension;
  double radial_tension;
};

// An items type using my face.
// A face type with a color member variable.
template <class Refs>
struct Cell_face : public CGAL::HalfedgeDS_face_base<Refs> {
  // Cell's life
  bool is_alive;
  int age;
  // Geometry
  double perimeter;
  double area;
  double volume;
  int num_sides;
  // Dynamical properties
  double contractility;
  double vol_elasticity;
};




struct Epithelium_items : public CGAL::Polyhedron_items_with_id_3 {
    template <class Refs, class Traits>
    struct Face_wrapper {
        typedef Cell_face<Refs> Face;
    };
    // template <class Refs, class Traits>
    // struct Edge_wrapper {
    //     typedef Junction_edge<Refs> Edge;
    // };
    // template <class Refs, class Traits>
    // struct Vertex_wrapper {
    //     typedef Junction_vertex<Refs> Vertex;
    // };
};
