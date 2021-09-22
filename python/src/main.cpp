#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <string>

#include "daisykitsdk/common/types.h"
#include "daisykitsdk/flows/background_matting_flow.h"
#include "daisykitsdk/flows/face_detector_flow.h"
#include "ndarray_converter.h"

using namespace daisykit;
namespace py = pybind11;

PYBIND11_MODULE(daisykit, m) {
  NDArrayConverter::init_numpy();

  py::class_<types::Face>(m, "Face")
      .def_readwrite("x", &types::Face::x)
      .def_readwrite("y", &types::Face::y)
      .def_readwrite("w", &types::Face::w)
      .def_readwrite("h", &types::Face::h)
      .def_readwrite("confidence", &types::Face::confidence)
      .def_readwrite("wearing_mask_prob", &types::Face::wearing_mask_prob)
      .def_readwrite("landmark", &types::Face::landmark);

  py::class_<types::Keypoint>(m, "Keypoint")
      .def_readwrite("x", &types::Keypoint::x)
      .def_readwrite("y", &types::Keypoint::y)
      .def_readwrite("confidence", &types::Keypoint::confidence);

  py::class_<flows::FaceDetectorFlow>(m, "FaceDetectorFlow")
      .def(py::init<const std::string&>(), py::arg("config_path"))
      .def("Process", &flows::FaceDetectorFlow::Process)
      .def("DrawResult", &flows::FaceDetectorFlow::DrawResult,
           py::return_value_policy::reference_internal);

  py::class_<flows::BackgroundMattingFlow>(m, "BackgroundMattingFlow")
      .def(py::init<const std::string&, const cv::Mat&>(),
           py::arg("config_path"), py::arg("default_background"))
      .def("Process", &flows::BackgroundMattingFlow::Process)
      .def("DrawResult", &flows::BackgroundMattingFlow::DrawResult,
           py::return_value_policy::reference_internal);

  m.doc() = R"pbdoc(
        DaisyKit python wrapper
        -----------------------
        .. currentmodule:: pydaisykit
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
