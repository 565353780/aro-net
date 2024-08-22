#pragma once

#include <MC.h>
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

  const bool detect(const std::vector<float> &points, const int &resolution,
                    const int &log_freq = 1);

  const bool toMeshFile(const std::string &save_mesh_file_path, const bool &overwrite = false);

  const bool detectAndSaveAsMeshFile(const std::vector<float> &points, const int &resolution,
      const std::string &save_mesh_file_path, const int &log_freq = 1, const bool &overwrite = false);

  const std::vector<float> getMeshVertices();
  const std::vector<int> getMeshFaces();

private:
  const torch::Tensor eval_points(const torch::Tensor &pcd,
                                  const torch::Tensor &qry,
                                  const torch::Tensor &anc,
                                  const int &log_freq = 1);

private:
  float padding_ = 0.1;
  float threshold_ = 0.5;
  int chunk_size_ = 3000;

  bool anchor_loaded_ = false;
  torch::Tensor anc_;

  torch::TensorOptions opts_ =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  bool use_gpu_ = false;

  std::shared_ptr<torch::jit::script::Module> model_ = nullptr;

  MC::mcMesh mesh_;
};
