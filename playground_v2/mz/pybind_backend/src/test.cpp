
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


py::array_t<float> add(py::array_t<float> input1, py::array_t<float> input2) {
  py::buffer_info buf1 = input1.request();
  py::buffer_info buf2 = input2.request();

  if (buf1.size != buf2.size) {
    throw std::runtime_error("Input shapes must match");
  }

  /*  allocate the buffer */
  py::array_t<float> result = py::array_t<float>(buf1.size);

  py::buffer_info buf3 = result.request();

  float *ptr1 = (float *) buf1.ptr,
         *ptr2 = (float *) buf2.ptr,
         *ptr3 = (float *) buf3.ptr;
  int X = buf1.shape[0];

  printf("HI %d", X);

  for (size_t idx = 0; idx < X; idx++) {
    ptr3[idx] = ptr1[idx] + ptr2[idx];
  }
 
  return result;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", &add, "A function that adds two numbers");
}