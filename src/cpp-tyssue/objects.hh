
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>



template <class Refs, class Point>
struct Junction_vertex : public CGAL::HalfedgeDS_vertex_base<Refs, CGAL::Tag_true, Point> {
public:
  bool get_active_state() const;
  void set_active_state(bool value);
private:
  bool is_active;
};

template <class Refs>
struct Junction_edge : public CGAL::HalfedgeDS_halfedge_base<Refs> {
public:
  double get_line_tension() const;
  void set_line_tension(double value);
  double get_radial_tension() const;
  void set_radial_tension(double value);
private:
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
  template <class Refs, class Traits>
  struct Halfedge_wrapper {
    typedef Junction_edge<Refs> Halfedge;
  };
  // template <class Refs, class Traits>
  // struct Vertex_wrapper {
  //   typedef typename Traits::Point_3 Point;
  //   typedef CGAL::HalfedgeDS_vertex_base<Refs, Point> Vertex;
  // };
};

typedef CGAL::Simple_cartesian<double>               Kernel;
typedef CGAL::Polyhedron_3<Kernel, Epithelium_items> Epithelium;
typedef Epithelium::Halfedge_handle                  Halfedge_handle;

//void make_hexagon(Epithelium &eptm){};
