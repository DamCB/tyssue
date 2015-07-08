#include <math.h>
// #include <CGAL/Linear_cell_complex.h>
#include <CGAL/Simple_cartesian.h>

#include <CGAL/Linear_cell_complex.h>
#include <CGAL/Combinatorial_map.h>
#include <CGAL/Combinatorial_map_constructors.h>
#include <CGAL/Combinatorial_map_operations.h>

#include "tyssue/objects.hh"

//long Uid::id;  // declares storage for static member Uid::id


#include <CGAL/Linear_cell_complex.h>
#include <CGAL/Linear_cell_complex_operations.h>
#include <CGAL/Linear_cell_complex_constructors.h>
#include <CGAL/Simple_cartesian.h>

typedef CGAL::Simple_cartesian<double>               Kernel;


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

void export_world(){
class_<World>("World")
  .def("greet", &World::greet)
  .def("set", &World::set);
}

void make_polygon(Appical_sheet_3 &sheet, std::vector<Point> &points) {
  std::size_t n_sides = points.size();
  Dart_handle dh = make_combinatorial_polygon(sheet, n_sides);
  Dart_handle prev = dh;
  Dart_handle next;
  for (std::vector<Point>::iterator it = points.begin(); it != points.end(); ++it){
    next = sheet.beta(prev, 1);
    Vertex_attribute_handle vh = sheet.create_vertex_attribute(*it);
    sheet.set_vertex_attribute(prev, vh);
    //int id = sheet.info<0>(vh);
    std::cout<<"Point: " <<sheet.point_of_vertex_attribute(vh)<<std::endl;
    prev = next;
  };
};

void make_hexagon(Appical_sheet_3 &sheet ) {
  Point p0, p1, p2, p3, p4, p5, p6;
  p0 = Point(0, 0, 0);
  p1 = Point(0, 1, 0);
  p2 = Point(0.5, sqrt(3)/2, 0);
  p3 = Point(0.5, -sqrt(3)/2, 0);
  p4 = Point(0, -1, 0);
  p5 = Point(-0.5, -sqrt(3)/2, 0);
  p6 = Point(0.5, -sqrt(3)/2, 0);

  std::vector<Point> points {p1, p2, p3, p4, p5, p6};
  make_polygon(sheet, points);
  // for (Appical_sheet_3::Vertex_attribute_range::iterator
  //      it=sheet.vertex_attributes().begin(),
  //      itend=sheet.vertex_attributes().end();
  //    it!=itend; ++it)
  //    {
  //      std::cout<<"point: "<<sheet.point_of_vertex_attribute(it)<<", "<<"id: "
  //           <<sheet.info_of_attribute<0>(it).id<<std::endl;
  //    }
};

double get_point_x(const Point point){
  return point.x();
}

double get_point_y(const Point point){
  return point.y();
}

double get_point_z(const Point point){
  return point.z();
}

void export_epithelium()
{
  def ("make_hexagon", make_hexagon);
  def ("make_polygon", make_polygon);
  //class_<Epithelium, bases<Poly> >("Epithelium")
  class_<Point>("Point", init<double, double, double>())
    .add_property("x", get_point_x)
    .add_property("y", get_point_y)
    .add_property("z", get_point_z)
    ;
  class_<Appical_sheet_3>("Epithelium")
    .def("is_valid", &Appical_sheet_3::is_valid)
    //    .def("barycenter", &Appical_sheet_3::barycenter<2>)
    ;
}
