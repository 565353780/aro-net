#include "mc_io.h"
#include <Eigen/src/Core/Matrix.h>
#include <filesystem>

const bool saveMeshFile(const MC::mcMesh &mesh,
                        const std::string &save_mesh_file_path,
                        const std::vector<float> &translate,
                        const float &scale,
                        const int &resolution,
                        const bool &overwrite){
  if (std::filesystem::exists(save_mesh_file_path)){
    if (!overwrite){
      return true;
    }

    std::filesystem::remove(save_mesh_file_path);
  }

  const float new_scale = scale / (resolution + 2);
  std::vector<float> new_translate{translate[0] - 0.5f, translate[1] - 0.5f, translate[2] - 0.5f};

  open3d::geometry::TriangleMesh o3d_mesh;

  o3d_mesh.vertices_.resize(mesh.vertices.size());
  for (int i = 0; i < mesh.vertices.size(); ++i) {
    o3d_mesh.vertices_[i] = Eigen::Vector3d(
        mesh.vertices[i].x, mesh.vertices[i].y, mesh.vertices[i].z);
  }

  o3d_mesh.vertex_normals_.resize(mesh.normals.size());
  for (int i = 0; i < mesh.normals.size(); ++i) {
    o3d_mesh.vertex_normals_[i] = Eigen::Vector3d(
        mesh.normals[i].x, mesh.normals[i].y, mesh.normals[i].z);
  }

  const int face_num = int(mesh.indices.size() / 3);

  o3d_mesh.triangles_.resize(face_num);
  for (int i = 0; i < face_num; ++i) {
    o3d_mesh.triangles_[i] = Eigen::Vector3i(
        mesh.indices[3 * i], mesh.indices[3 * i + 1], mesh.indices[3 * i + 2]);
  }

  Eigen::Matrix3d rotate_matrix;
  rotate_matrix <<
    std::cos(M_PI_2), 0.0, -std::sin(M_PI_2),
    0.0, 1.0, 0.0,
    std::sin(M_PI_2), 0.0, std::cos(M_PI_2);

  o3d_mesh.Translate(Eigen::Vector3d(-1.0, -1.0, -1.0));
  o3d_mesh.Scale(1.0 / resolution, Eigen::Vector3d(0.0, 0.0, 0.0));
  o3d_mesh.Translate(Eigen::Vector3d(-0.5, -0.5, -0.5));
  o3d_mesh.Rotate(rotate_matrix, Eigen::Vector3d(0.0, 0.0, 0.0));
  o3d_mesh.Scale(scale, Eigen::Vector3d(0.0, 0.0, 0.0));
  o3d_mesh.Translate(Eigen::Vector3d(translate[0], translate[1], translate[2]));

  const std::string save_mesh_folder_path = std::filesystem::path(save_mesh_file_path).parent_path();
  if (!std::filesystem::exists(save_mesh_folder_path)) {
    std::filesystem::create_directories(save_mesh_folder_path);
  }

  open3d::io::WriteTriangleMesh(save_mesh_file_path, o3d_mesh);

  return true;
}
