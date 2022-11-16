#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <iostream>

int hello_world() {
    std::cout << "Hello World!";
    return 0;
}

PYBIND11_MODULE(hello, m) {
    m.def("hello_world", &hello_world);
}