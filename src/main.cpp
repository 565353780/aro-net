#include "detector.h"
#include <iostream>

int main() {
  // super params
  const std::string anchor_file_path =
      "/home/chli/github/AMCAX/point-cloud-reverse-algorithm/aro-net/data/anchors/anc_48.npy";
  const std::string model_file_path =
      "/home/chli/github/AMCAX/point-cloud-reverse-algorithm/aro-net/output/aronet_cpp.pt";
  const bool use_gpu = true;
  const int resolution = 32;
  const std::string save_mesh_file_path = "./output/recon_aro.ply";
  const int log_freq = 1;
  const bool overwrite = false;

  // input point cloud [x1, y1, z1, x2, y2, z2, ...]
  std::vector<float> points;
  points.resize(300);
  for (int i = 0; i < 100; ++i) {
    points[3 * i] = 1.0 * i;
    points[3 * i + 1] = 2.0 * i;
    points[3 * i + 2] = 3.0 * i;
  }

  // construct detector module
  Detector detector(anchor_file_path, model_file_path, use_gpu);

  if (!detector.isValid()) {
    std::cout << "init detector failed!" << std::endl;
    return -1;
  }

  // reconstruct mesh from input point cloud
  const bool success = detector.detectAndSaveAsMeshFile(points, resolution, save_mesh_file_path, log_freq, overwrite);
  if (!success) {
    std::cout << "detectAndSaveAsMeshFile failed!" << std::endl;
    return -1;
  }

  // get reconstructed mesh data
  const std::vector<float> vertices = detector.getMeshVertices();
  const std::vector<int> faces = detector.getMeshFaces();

  std::cout << "vertices num: " << int(vertices.size() / 3) << std::endl;
  std::cout << "faces num: " << int(faces.size() / 3) << std::endl;

  // clear loaded model
  detector.clear();

  return 1;
}
