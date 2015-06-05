#include <CGAL/Linear_cell_complex.h>

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

void export_python()
{
  class_<World>("World", no_init)
    .def("greet", &World::greet)
    .def("set", &World::set);
}
