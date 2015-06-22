#include <CGAL/Linear_cell_complex.h>
#include <CGAL/Linear_cell_complex_operations.h>
#include <CGAL/Linear_cell_complex_constructors.h>
#include <CGAL/Simple_cartesian.h>

typedef CGAL::Simple_cartesian<double>               Kernel;

struct Uid {
public:
  long new_id() {
    id +=1;
    return id;
}
  void reset() {id = 0;}
private:
  static long id
};

struct Vid::Uid {};

struct Eid::Uid {};

struct Cid::Uid {};

struct Vertex_data {
  long id = Vid.new_id();
}

struct Junction_data {
  long id = Eid.new_id();
  typedef Kernel::Vector_3 Vector;
  Vector gradient;
  double length;
  double line_tension;
};

struct Cell_data {
  long id = Cid.new_id();
  double perimeter;
  double area;
  double volume;
  double prefered_volume;
  double contractility;
  double vol_elasticity;
};

struct Average_functor
{
  template<class CellAttribute>
  void operator()(CellAttribute& ca1, CellAttribute& ca2)
  { ca1.info()=(ca1.info()+ ca2.info())/2; }
};

struct Vertex_functor
{
  template<class CellAttribute>
  void operator()(CellAttribute& ca1, CellAttribute& ca2)
  { ca2.info().id = Cid.new_id(); }
};

struct Junction_merge_functor
{
  template<class CellAttribute>
  void operator()(CellAttribute& ca1, CellAttribute& ca2)
  {
    ca1.info().gradient=(ca1.info().gradient + ca2.info().gradient)/2;
    ca1.info().length=(ca1.info().length + ca2.info().length)/2;
    ca1.info().line_tension=(ca1.info().line_tension + ca2.info().line_tension)/2;
 }
};

struct Junction_split_functor
{
  template<class CellAttribute>
  void operator()(CellAttribute& ca1, CellAttribute& ca2)
  {
    //ca2.info().id = Cid.new_id();
    ca1.info().gradient= ca1.info().gradient/2.;
    ca2.info().gradient= ca2.info().gradient/2.;
    ca1.info().length= ca1.info().length/2;
    ca2.info().length= ca1.info().length/2;
    ca1.info().line_tension= ca1.info().line_tension;
    ca2.info().line_tension= ca1.info().line_tension;
 }
};


struct Cell_merge_functor
{
  template<class CellAttribute>
  void operator()(CellAttribute& ca1, CellAttribute& ca2)
  {
    ca1.info().perimeter=(ca1.info().perimeter + ca2.info().perimeter);
    ca1.info().area=(ca1.info().area + ca2.info().area);
    ca1.info().volume=(ca1.info().volume + ca2.info().volume);
    ca1.info().prefered_volume=(ca1.info().prefered_volume
				+ ca2.info().prefered_volume)/2;
    ca1.info().contractility=(ca1.info().contractility + ca2.info().contractility)/2;
    ca1.info().vol_elasticity=(ca1.info().vol_elasticity + ca2.info().vol_elasticity)/2;
 }
};

struct Cell_split_functor
{
  template<class CellAttribute>
  void operator()(CellAttribute& ca1, CellAttribute& ca2)
  {
    ca1.info().perimeter = ca1.info().perimeter/2;
    ca2.info().perimeter = ca1.info().perimeter/2;
    ca1.info().area = ca1.info().area/2;
    ca2.info().area = ca1.info().area/2;
    ca1.info().volume = ca1.info().volume/2;
    ca2.info().volume = ca1.info().volume/2;
    ca1.info().contractility = ca1.info().contractility;
    ca2.info().contractility = ca1.info().contractility;
    ca1.info().vol_elasticity = ca1.info().vol_elasticity;
    ca2.info().vol_elasticity = ca1.info().vol_elasticity;
 }
};




struct Epithelium_Items
{
  template<class Refs>
  struct Dart_wrapper
  {
    typedef CGAL::Dart<2, Refs > Dart;
    // Vertex attribute's info switch between active/inactive
    typedef CGAL::Cell_attribute_with_point< Refs, bool, CGAL::Tag_true,
                                             Vertex_functor >
    Vertex_attribute;
    typedef CGAL::Cell_attribute< Refs, Junction_data, CGAL::Tag_true,
				  Junction_merge_functor,
				  Junction_split_functor >
    Edge_attribute;

    typedef CGAL::Cell_attribute< Refs, Cell_data, CGAL::Tag_true,
				  Cell_merge_functor,
				  Cell_split_functor >
    Celldart_attribute;

    typedef CGAL::cpp11::tuple<Vertex_attribute, Edge_attribute, Celldart_attribute>
    Attributes;
  };
};

typedef CGAL::Linear_cell_complex_traits<3, Kernel>                Traits;
typedef CGAL::Linear_cell_complex<2,3,Traits,Epithelium_Items>     Appical_sheet_3;
typedef Appical_sheet_3::Dart_handle                               Dart_handle;
typedef Appical_sheet_3::Point                                     Point;
typedef Appical_sheet_3::FT                                        FT;
typedef Appical_sheet_3::Vertex_attribute                          Vertex_attribute;
typedef Appical_sheet_3::Vertex_attribute_handle                   Vertex_attribute_handle;
