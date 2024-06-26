#include "detector.h"
#include <iostream>

int main() {
  // super params
  const std::string root_path = "../../aro-net/";
  const std::string model_file_path = "../aro-net/output/4_7080.ckpt";

  // input point cloud [x1, y1, z1, x2, y2, z2, ...]
  std::vector<float> points;
  points.resize(3000);
  for (int i = 0; i < 1000; ++i) {
    points[3 * i] = 1.0 * i;
    points[3 * i + 1] = 2.0 * i;
    points[3 * i + 2] = 3.0 * i;
  }

  // construct detector module
  Detector detector(root_path, model_file_path);

  // reconstruct mesh from input point cloud
  const bool success = detector.reconMesh(points);
  if (!success) {
    std::cout << "reconMesh failed!" << std::endl;
    return -1;
  }

  // get reconstructed mesh data
  const std::vector<float> vertices = detector.getVertices();
  const std::vector<float> faces = detector.getFaces();

  std::cout << "vertices num: " << int(vertices.size() / 3) << std::endl;
  std::cout << "faces num: " << int(faces.size() / 3) << std::endl;

  return 1;
}
