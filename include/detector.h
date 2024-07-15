#pragma once

#include <torch/extension.h>

class Detector {
public:
  Detector(){};
  Detector(const std::string &anchor_file_path,
           const std::string &model_file_path, const bool &use_gpu = false);

public:
  const bool clear();

  const bool isValid();

  const bool loadAnchors(const std::string &anchor_file_path);

  const bool loadModel(const std::string &model_file_path);

private:
  bool anchor_loaded_ = false;
  torch::Tensor anc_;

  torch::Device device_ = torch::kCPU;

  std::shared_ptr<torch::jit::script::Module> model_ = nullptr;
};
