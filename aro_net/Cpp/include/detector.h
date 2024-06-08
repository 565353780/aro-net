#pragma once

#include <pybind11/embed.h>
#include <string>

namespace py = pybind11;

class __attribute__((visibility("default"))) Detector {
public:
  Detector(const std::string &root_path = "../../aro-net/",
           const std::string &model_file_path = "");
  ~Detector();

  const bool reconMesh(const std::vector<float> &points);

  const std::vector<float> getVertices();
  const std::vector<float> getFaces();

private:
  py::scoped_interpreter guard_{};

  py::object detector_;
};
