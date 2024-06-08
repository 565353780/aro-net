#include "detector.h"
#include <vector>

using namespace pybind11::literals;

Detector::Detector(const std::string &root_path,
                   const std::string &model_file_path) {
  py::gil_scoped_acquire acquire;

  py::object sys = py::module_::import("sys");

  sys.attr("path").attr("append")(root_path);

  py::object Detector = py::module_::import("aro_net.Module.detector");

  detector_ = Detector.attr("Detector")("model_file_path"_a = model_file_path);

  return;
}

Detector::~Detector() {}

const bool Detector::reconMesh(const std::vector<float> &points) {
  py::gil_scoped_acquire acquire;

  py::list point_list;
  for (int i = 0; i < points.size(); ++i) {
    point_list.append(points[i]);
  }

  const bool success =
      detector_.attr("detectPointsList")("points"_a = point_list).cast<bool>();

  return success;
}

const std::vector<float> Detector::getVertices() {
  py::list vertices_list = detector_.attr("getVerticesList")();

  std::vector<float> vertices;
  vertices.reserve(vertices_list.size());

  for (int i = 0; i < vertices_list.size(); ++i) {
    vertices.emplace_back(vertices_list[i].cast<float>());
  }

  return vertices;
}

const std::vector<float> Detector::getFaces() {
  py::list faces_list = detector_.attr("getFacesList")();

  std::vector<float> faces;
  faces.reserve(faces_list.size());

  for (int i = 0; i < faces_list.size(); ++i) {
    faces.emplace_back(faces_list[i].cast<float>());
  }

  return faces;
}
