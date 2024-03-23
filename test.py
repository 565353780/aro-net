import open3d as o3d
import numpy as np

# the data director of pcd.np
pcd_np_files=["/home/js/zz/ARO-Net/data/shapenet/01_pcds/02691156/occnet/1a04e3eab45ca15dd86060f189eb133.npy",\
              "/home/js/zz/ARO-Net/data/shapenet/02_qry_pts_imnet/02691156/1a04e3eab45ca15dd86060f189eb133.npy",\
              "/home/js/zz/ARO-Net/data/shapenet/02_qry_pts_occnet/02691156/1a04e3eab45ca15dd86060f189eb133.npy",\
              ]


pcd_list=[]

for pcd_np_file in pcd_np_files: 
    pcd = o3d.geometry.PointCloud() 
    point_cloud_data=np.load(pcd_np_file)
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data)
    pcd_list.append(pcd)


# 保存点云到 .pcd 文件
# o3d.io.write_point_cloud("output_point_cloud_file.pcd", pcd)
pcd = o3d.io.read_triangle_mesh('/home/js/zz/ARO-Net/experiments/pretrained_chairs/results/airplane/airplane4.obj')
# pcd = o3d.io.read_point_cloud('/home/js/airplane.ply')
pcd_data=np.load('/home/js/zz/ARO-Net/data/shapenet/01_pcds/02691156/2048/d1a8e79eebf4a0b1579c3d4943e463ef.npy')
pcd1 = o3d.geometry.PointCloud() 
pcd1.points=o3d.utility.Vector3dVector(pcd_data)
# 创建一个 Open3D Visualizer
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()

# sub_windows=[]

# 添加点云到 Visualizer
# for pcd in pcd_list:
#     # sub_window=visualizer.create_window()
#     # sub_windows.append(sub_window)
#     visualizer.add_geometry(pcd)
    # o3d.visualization.draw_geometries([pcd],window_name=f"Sub Window {i}", width=800, height=600, window=sub_window)

visualizer.add_geometry(pcd)
visualizer.run()   
