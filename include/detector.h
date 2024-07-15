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

  const bool detect(const std::vector<float> &points, const int &resolution);

private:
  const torch::Tensor eval_points(const torch::Tensor &pcd,
                                  const torch::Tensor &qry,
                                  const torch::Tensor &anc);

private:
  float padding_ = 0.1;
  float threshold_ = 0.5;
  int chunk_size_ = 3000;

  bool anchor_loaded_ = false;
  torch::Tensor anc_;

  torch::TensorOptions opts_ =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  std::shared_ptr<torch::jit::script::Module> model_ = nullptr;
};
