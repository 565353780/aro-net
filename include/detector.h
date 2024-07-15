#pragma once

#include <torch/extension.h>

class Detector {
public:
  Detector(){};
  Detector(const std::string &model_file_path);

public:
  const bool isValid();

  const bool loadModel(const std::string &model_file_path);

private:
  bool model_loaded_ = false;

  std::shared_ptr<torch::jit::script::Module> model_;
};
