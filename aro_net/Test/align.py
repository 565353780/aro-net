import numpy as np
import open3d as o3d

def test():
    denoised_pcd_file_path = '../pipeline/output/denoise_pcd.ply'
    aro_mesh_file_path = '../pipeline/output/recon_aro.ply'

    pcd = o3d.io.read_point_cloud(denoised_pcd_file_path)
    mesh = o3d.io.read_triangle_mesh(aro_mesh_file_path)

    translate = [0.00138721, 0.0019809, 0.0013134]
    scale = 0.903143
    resolution = 64

    rotate_rad = np.pi / 2.0

    rotate_matrix = np.asarray([
        [np.cos(rotate_rad), 0.0, -np.sin(rotate_rad)],
        [0.0, 1.0, 0.0],
        [np.sin(rotate_rad), 0.0, np.cos(rotate_rad)],
    ])

    mesh.translate(-1 * np.ones(3))
    mesh.scale(1.0 / resolution, np.asarray([0.0 for _ in range(3)]))
    mesh.translate(-0.5 * np.ones(3))
    mesh.rotate(rotate_matrix)
    mesh.scale(scale, np.asarray([0.0 for _ in range(3)]))
    mesh.translate(1.0 * np.asarray(translate))

    o3d.visualization.draw_geometries([pcd, mesh])
    return True
