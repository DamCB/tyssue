#include <math.h>
#include <CGAL/Simple_cartesian.h>

#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>

#include <CGAL/Linear_cell_complex.h>
#include <CGAL/Combinatorial_map_constructors.h>

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

void export_world(){
class_<World>("World")
  .def("greet", &World::greet)
  .def("set", &World::set);
}

void make_hexagon(Appical_sheet_3 &sheet ) {
  Point p0, p1, p2, p3, p4, p5, p6;
  p0 = Point(0, 0, 0);
  p1 = Point( 0, 1, 0);
  p2 = Point( 0.5, sqrt(3)/2, 0);
  p3 = Point( 0.5, -sqrt(3)/2, 0);
  p4 = Point( 0, -1, 0);
  p5 = Point( -0.5, -sqrt(3)/2, 0);
  p6 = Point( 0.5, -sqrt(3)/2, 0);
  Dart_handle dh01 = sheet.make_triangle(p0, p1, p2);

  Dart_handle dh12 = sheet.beta(dh01, 1);
  Dart_handle dh20 = sheet.beta(dh12, 1);
  Dart_handle dh02 = sheet.make_triangle(p0, p2, p3);
  sheet.sew<1>(dh20, dh02);
  Dart_handle dh23 = sheet.beta(dh02, 1);
  Dart_handle dh30 = sheet.beta(dh23, 1);
  Dart_handle dh03 = sheet.make_triangle(p0, p3, p4);
  sheet.sew<1>(dh30, dh03);
  Dart_handle dh34 = sheet.beta(dh03, 1);
  Dart_handle dh40 = sheet.beta(dh34, 1);
  Dart_handle dh04 = sheet.make_triangle(p0, p4, p5);
  sheet.sew<1>(dh40, dh04);
  Dart_handle dh45 = sheet.beta(dh04, 1);
  Dart_handle dh50 = sheet.beta(dh45, 1);
  Dart_handle dh05 = sheet.make_triangle(p0, p5, p6);
  sheet.sew<1>(dh50, dh05);
  Dart_handle dh56 = sheet.beta(dh05, 1);
  Dart_handle dh60 = sheet.beta(dh56, 1);
  Dart_handle dh06 = sheet.make_triangle(p0, p6, p1);
  sheet.sew<1>(dh60, dh06);
  Dart_handle dh61 = sheet.beta(dh06, 1);
  Dart_handle dh10 = sheet.beta(dh61, 1);
  sheet.sew<1>(dh10, dh01);
};
// void (Epithelium &eptm){
//   Build_hexagon<HalfedgeDS> hexagon;
//   eptm.delegate( hexagon);
// };



void export_epithelium()
{
  def ("make_hexagon", make_hexagon);
  //class_<Epithelium, bases<Poly> >("Epithelium")
  class_<Appical_sheet_3>("Epithelium")
    ;
}
