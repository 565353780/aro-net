#define MC_IMPLEM_ENABLE
#include "detector.h"
#include "cnpy.h"
#include "grid.h"
#include "mc_io.h"
#include "transform.h"
#include <ATen/ops/from_blob.h>
#include <filesystem>
#include <memory>
#include <open3d/Open3D.h>
#include <open3d/io/TriangleMeshIO.h>
#include <torch/script.h>
#include <torch/types.h>

using namespace torch::indexing;

Detector::Detector(const std::string &anchor_file_path,
                   const std::string &model_file_path, const bool &use_gpu) {
  if (use_gpu) {
    opts_ = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  } else {
    opts_ = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  }
  use_gpu_ = use_gpu;

  if (!loadAnchors(anchor_file_path)) {
    std::cout << "[ERROR][Detector::Detector]" << std::endl;
    std::cout << "\t loadAnchors failed!" << std::endl;
  }

  if (!loadModel(model_file_path)) {
    std::cout << "[ERROR][Detector::Detector]" << std::endl;
    std::cout << "\t loadModel failed!" << std::endl;
  }
}

const bool Detector::clear() {
  model_.reset();

  mesh_.vertices.clear();
  mesh_.normals.clear();
  mesh_.indices.clear();

  return true;
}

const bool Detector::isValid() {
  if (model_ == nullptr) {
    return false;
  }

  if (!anchor_loaded_) {
    return false;
  }

  return true;
}

const bool Detector::loadAnchors(const std::string &anchor_file_path) {
  anchor_loaded_ = false;

  if (!std::filesystem::exists(anchor_file_path)) {
    std::cout << "[ERROR][Detector::loadAnchors]" << std::endl;
    std::cout << "\t anchor file not exist!" << std::endl;
    std::cout << "\t anchor_file_path: " << anchor_file_path << std::endl;

    return false;
  }

  cnpy::NpyArray arr = cnpy::npy_load(anchor_file_path);

  const int anchor_size = arr.shape[0] * arr.shape[1];

  anc_ = torch::from_blob(arr.data<float>(), {long(anchor_size)}, opts_)
             .reshape({1, long(arr.shape[0]), long(arr.shape[1])})
             .detach()
             .clone();
  if (use_gpu_){
    anc_ = anc_.cuda();
  }

  anchor_loaded_ = true;
  return true;
  ;
}

const bool Detector::loadModel(const std::string &model_file_path) {
  if (!std::filesystem::exists(model_file_path)) {
    std::cout << "[ERROR][Detector::loadModel]" << std::endl;
    std::cout << "\t model file not exist!" << std::endl;
    std::cout << "\t model_file_path: " << model_file_path << std::endl;

    return false;
  }

  model_ = std::make_shared<torch::jit::script::Module>(
      torch::jit::load(model_file_path));
  if (use_gpu_){
    model_->to(torch::kCUDA);
  }
  model_->eval();

  return true;
}

const bool Detector::detect(const std::vector<float> &points,
                            const int &resolution, const int &log_freq) {
  mesh_.vertices.clear();
  mesh_.normals.clear();
  mesh_.indices.clear();

  const float threshold = std::log(threshold_) - std::log(1.0 - threshold_);

  const float box_size = 1.0f + padding_;

  const torch::Tensor pointsf =
      box_size * make_3d_grid({-0.5, -0.5, -0.5}, {0.5, 0.5, 0.5},
                              {resolution, resolution, resolution});

  torch::Tensor qry = pointsf.unsqueeze(0).to(opts_);
  if (use_gpu_){
    qry = qry.cuda();
  }

  torch::Tensor pcd =
      torch::from_blob((void *)(points).data(), {long(points.size())}, opts_)
          .reshape({-1, 3});
  if (use_gpu_){
    pcd = pcd.cuda();
  }

  torch::Tensor translate;
  float scale;
  normalizePoints(pcd, translate, scale);

  const torch::Tensor trans_pcd = ((pcd - translate) / scale).unsqueeze(0);

  const torch::Tensor values = eval_points(trans_pcd, qry, anc_, log_freq);

  const torch::Tensor value_grid =
      values.reshape({resolution, resolution, resolution});

  torch::Tensor occ_hat_padded = torch::nn::functional::pad(
      value_grid, torch::nn::functional::PadFuncOptions({1, 1, 1, 1, 1, 1})
                      .mode(torch::kConstant)
                      .value(-1e6));

  const torch::Tensor safe_occ_hat_padded = occ_hat_padded.reshape({-1}).detach().clone().cpu();

  MC::MC_FLOAT *field = safe_occ_hat_padded.data_ptr<float>();

  MC::marching_cube(field, resolution + 2, resolution + 2, resolution + 2,
                    mesh_);

  return true;
}

const bool Detector::toMeshFile(const std::string &save_mesh_file_path) {
  if (!saveMeshFile(mesh_, save_mesh_file_path)) {
    std::cout << "[ERROR][Detector::toMeshFile]" << std::endl;
    std::cout << "\t saveMeshFile failed!" << std::endl;

    return false;
  }

  return true;
}

const bool Detector::detectAndSaveAsMeshFile(const std::vector<float> &points, const int &resolution,
      const std::string &save_mesh_file_path, const int &log_freq, const bool &overwrite){
  if (std::filesystem::exists(save_mesh_file_path)){
    if (!overwrite){
      return true;
    }

    std::filesystem::remove(save_mesh_file_path);
  }

  if (!detect(points, resolution, log_freq)){
    std::cout << "[ERROR][Detector::detectAndSaveAsMeshFile]" << std::endl;
    std::cout << "\t detect failed!" << std::endl;
    return false;
  }

  if (!toMeshFile(save_mesh_file_path)){
    std::cout << "[ERROR][Detector::detectAndSaveAsMeshFile]" << std::endl;
    std::cout << "\t toMeshFile failed!" << std::endl;
    return false;
  }

  return true;
}

const std::vector<float> Detector::getMeshVertices() {
  std::vector<float> recon_vertices(mesh_.vertices.size() * 3);
  for (int i = 0; i < mesh_.vertices.size(); ++i) {
    recon_vertices[3 * i] = mesh_.vertices[i].x;
    recon_vertices[3 * i + 1] = mesh_.vertices[i].y;
    recon_vertices[3 * i + 2] = mesh_.vertices[i].z;
  }

  return recon_vertices;
}

const std::vector<int> Detector::getMeshFaces() {
  std::vector<int> recon_faces(mesh_.indices.size());
  for (int i = 0; i < mesh_.indices.size(); ++i) {
    recon_faces[i] = mesh_.indices[i];
  }

  return recon_faces;
}

const torch::Tensor Detector::eval_points(const torch::Tensor &pcd,
                                          const torch::Tensor &qry,
                                          const torch::Tensor &anc,
                                          const int &log_freq) {
  torch::NoGradGuard no_grad;

  const int n_qry = qry.size(1);

  int chunk_size;
  if (use_gpu_){
    chunk_size = chunk_size_;
  }
  else{
    chunk_size = int(chunk_size_ * 0.1);
  }

  const int n_chunk = std::ceil(n_qry / chunk_size);

  std::vector<torch::Tensor> ret;

  std::vector<torch::jit::IValue> inputs;
  inputs.resize(3);
  inputs[0] = pcd;
  inputs[2] = anc;

  std::cout << "[INFO][Detector::eval_points]" << std::endl;
  std::cout << "\t start predict occ field..." << std::endl;
  for (int i = 0; i < n_chunk; ++i) {
    torch::Tensor qry_chunk;
    if (i < n_chunk - 1) {
      qry_chunk = qry.index(
          {Slice(), Slice(chunk_size * i, chunk_size * (i + 1)), Ellipsis});
    } else {
      qry_chunk = qry.index({Slice(), Slice(chunk_size * i, n_qry), Ellipsis});
    }

    inputs[1] = qry_chunk;

    const torch::Tensor occ = model_->forward(inputs).toTensor();

    ret.emplace_back(occ);

    if (i % log_freq == 0) {
      std::cout << "\r [INFO][Detector::eval_points] occ predict: " << i + 1
                << " / " << n_chunk << "...    ";
    }
  }
  std::cout << std::endl;

  const torch::Tensor result = torch::cat(ret, -1).squeeze(0);

  return result;
}
