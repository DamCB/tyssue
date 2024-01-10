#include <pybind11/pybind11.h>

namespace py = pybind11;

size_t free_function() {
      return 42;
}


PYBIND11_MODULE(tyssue_cpp, m)
{

  m.doc() = "This is top module - mymodule.";
  m.def("free_function", &free_function);
}
