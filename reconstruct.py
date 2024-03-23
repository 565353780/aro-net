import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange, tqdm
import trimesh
from src_convonet.utils import libmcubes
from src_convonet.common import make_3d_grid, normalize_coord, add_key, coord2index
from src_convonet.utils.libsimplify import simplify_mesh
from src_convonet.utils.libmise import MISE
import time
import math
from src import datasets
from src.models import ARONetModel
from src.datasets import ARONetDataset
import os
import open3d as o3d
from options import get_parser

class Generator3D(object):
    '''  Generator class for Occupancy Networks.
    It provides functions to generate the final mesh as well refining options.
    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        input_type (str): type of input
        vol_info (dict): volume infomation
        vol_bound (dict): volume boundary
        simplify_nfaces (int): number of faces the mesh should be simplified to
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution0=64, upsampling_steps=2, chunk_size=3000,
                 with_normals=False, padding=0.1, sample=False,
                 input_type = None,
                 vol_info = None,
                 vol_bound = None,
                 simplify_nfaces=None, pred_type='occ'):
        #self.model = model.to(device)
        self.model = model
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.input_type = input_type
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces
        self.chunk_size = chunk_size
        self.pred_type = pred_type
        # for pointcloud_crop
        self.vol_bound = vol_bound
        if vol_info is not None:
            self.input_vol, _, _ = vol_info



    def eval_points(self, data):
        
        n_qry = data['qry'].shape[1]
        chunk_size = self.chunk_size
        n_chunk = math.ceil(n_qry / chunk_size)
        
        ret = []

        for idx in range(n_chunk):
            data_chunk = {}
            for key in data:
                if key == 'qry':
                    if idx < n_chunk - 1:
                        data_chunk[key] = data[key][:, chunk_size*idx:chunk_size*(idx+1), ...]
                    else:
                        data_chunk[key] = data[key][:, chunk_size*idx:n_qry, ...]
                else:
                    data_chunk[key] =  data[key]
            ret_dict = self.model(data_chunk)#have some problem
            if self.pred_type == 'occ':
                ret.append(ret_dict['occ_pred'])
            else:
                ret.append(ret_dict['sdf_pred'])

        
        ret = torch.cat(ret, -1)
        ret = ret.squeeze(0)
        return ret

    def generate_mesh(self, data, return_stats=True):
        ''' Generates the output mesh.
        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        # self.model.eval()
        device = self.device
        stats_dict = {}
        mesh = self.generate_from_latent(data, stats_dict=stats_dict)

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh
    
    def generate_from_latent(self, c=None, stats_dict={}):
        ''' Generates mesh from latent.
            Works for shapes normalized to a unit cube
        Args:
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding
        
        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )
            data['qry'] = pointsf.unsqueeze(0).cuda()
            values = self.eval_points(data).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(
                8, self.upsampling_steps, threshold)
            
            points = mesh_extractor.query()
            while points.shape[0] != 0:
                # Query points
                pointsf = points / mesh_extractor.resolution
                # Normalize to bounding box
                pointsf = box_size * (pointsf - 0.5)
                pointsf = torch.FloatTensor(pointsf).to(self.device)
                data['qry'] = pointsf.unsqueeze(0).cuda()
                # Evaluate model and update
                values = self.eval_points(data).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0
        

        mesh = self.extract_mesh(value_grid, c, stats_dict=stats_dict)
        return mesh

    def extract_mesh(self, occ_hat, c=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.
        Args:
            occ_hat (tensor): value grid of occupancies
            c (tensor): encoded feature volumes
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # # Undo padding
        vertices -= 1
        
        if self.vol_bound is not None:
            # Scale the mesh back to its original metric
            bb_min = self.vol_bound['query_vol'][:, 0].min(axis=0)
            bb_max = self.vol_bound['query_vol'][:, 1].max(axis=0)
            mc_unit = max(bb_max - bb_min) / (self.vol_bound['axis_n_crop'].max() * self.resolution0*2**self.upsampling_steps)
            vertices = vertices * mc_unit + bb_min
        else: 
            # Normalize to bounding box
            vertices /= np.array([n_x-1, n_y-1, n_z-1])
            vertices = box_size * (vertices - 0.5)
        
        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, c)
            stats_dict['time (normals)'] = time.time() - t0

        else:
            normals = None


        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)
        


        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh

    def estimate_normals(self, vertices, c=None):
        ''' Estimates the normals by computing the gradient of the objective.
        Args:
            vertices (numpy array): vertices of the mesh
            c (tensor): encoded feature volumes
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        c = c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decode(vi, c).logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, c=None):
        ''' Refines the predicted mesh.
        Args:   
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.model.decode(face_point.unsqueeze(0), c).logits
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh

# if __name__ == '__main__':
#     args = get_parser().parse_args()

#     model = ARONetModel(n_anc=args.n_anc, n_qry=args.n_qry, n_local=args.n_local, cone_angle_th=args.cone_angle_th, tfm_pos_enc=args.tfm_pos_enc, 
#                         cond_pn=args.cond_pn, use_dist_hit=args.use_dist_hit, pn_use_bn=args.pn_use_bn, pred_type=args.pred_type, norm_coord=args.norm_coord)
#     path_ckpt = os.path.join('experiments', args.name_exp, 'ckpt', args.name_ckpt)
#     model.load_state_dict(torch.load(path_ckpt)['model'])
#     model = model.cuda()
#     model = model.eval()
#     # create the results folder to save the results
#     path_res = os.path.join('experiments', args.name_exp, 'results', args.name_dataset)
#     if not os.path.exists(path_res):
#         os.makedirs(path_res)

#     generator = Generator3D(model, threshold=args.mc_threshold, resolution0=args.mc_res0, upsampling_steps=args.mc_up_steps, 
#                             chunk_size=args.mc_chunk_size, pred_type=args.pred_type)
#     dataset = ARONetDataset(split='test', args=args)
    
#     dir_dataset = os.path.join(args.dir_data, args.name_dataset)
#     if args.name_dataset == 'shapenet':
#         categories = args.categories_test.split(',')[:-1]
#         id_shapes = []
#         for category in categories:
#             id_shapes_ = open(f'{dir_dataset}/04_splits/{category}/test.lst').read().split('\n')
#             id_shapes += id_shapes_

#     else:
#         id_shapes = open(f'{dir_dataset}/04_splits/test.lst').read().split('\n')
#     with torch.no_grad():
        
#         for idx in tqdm(range(len(dataset))):
         
#             data = dataset[idx]
#             # print('data', data)
#             for key in data:
#                 data[key] = data[key].unsqueeze(0).cuda() 
#             out = generator.generate_mesh(data)
#             try:
#                 mesh, stats_dict = out
#             except TypeError:
#                 mesh, stats_dict = out, {}
                
#             path_mesh = os.path.join(path_res, '%s.obj'%id_shapes[idx])
#             mesh.export(path_mesh)

if __name__ == '__main__':
    args = get_parser().parse_args()

    model = ARONetModel(n_anc=args.n_anc, n_qry=args.n_qry, n_local=args.n_local, cone_angle_th=args.cone_angle_th, tfm_pos_enc=args.tfm_pos_enc, 
                        cond_pn=args.cond_pn, use_dist_hit=args.use_dist_hit, pn_use_bn=args.pn_use_bn, pred_type=args.pred_type, norm_coord=args.norm_coord)
    path_ckpt = os.path.join('experiments', args.name_exp, 'ckpt', args.name_ckpt)
    model.load_state_dict(torch.load(path_ckpt)['model'])
    model = model.cuda()
    model = model.eval()

    generator = Generator3D(model, threshold=args.mc_threshold, resolution0=args.mc_res0, upsampling_steps=args.mc_up_steps, 
                            chunk_size=args.mc_chunk_size, pred_type=args.pred_type)
    
    anc_0 = np.load(f'./{args.dir_data}/anchors/sphere{str(args.n_anc)}.npy')
    anc = np.concatenate([anc_0[i::3] / (2 ** i) for i in range(3)])
    
    pcd_ply=o3d.io.read_point_cloud("/home/js/airplane_pcd.ply")
    pcd_ply=pcd_ply.farthest_point_down_sample(250000)
    # bbox = pcd_ply.get_axis_aligned_bounding_box()
    
    # # 获取边界框的中心和尺寸
    # pcd_ply.translate(bbox.get_center())
    # size = bbox.get_max_bound() - bbox.get_min_bound()
    # pcd_ply.scale(1.0/max(size),center=pcd_ply.get_center())
    
   
    pcd=np.asarray(pcd_ply.points)
    # pcd=np.load("/home/js/zz/ARO-Net/data/shapenet/01_pcds/02691156/2048/d1a8e79eebf4a0b1579c3d4943e463ef.npy")
    
    print(len(pcd))
    bounds = np.ptp(pcd, axis=0)

    # Check if the point cloud is already normalized
    if np.min(bounds) != 0.0:
        translation = -np.mean(pcd, axis=0)
        pcd += translation
        scale = 1.0 / np.max(bounds)
        pcd*= scale

    data={
            'pcd': torch.tensor(pcd).float(),
            # 'qry': torch.tensor(qry).float(),
            'anc': torch.tensor(anc).float(),
            # 'occ': torch.tensor(occ).float(),
            # 'sdf': torch.tensor(sdf).float(),
        }
    # data = ARONetDataset(split='test', args=args)[0]  # 只取第一个样本
    print(data['pcd'])
    for key in data:
        data[key] = data[key].unsqueeze(0).cuda()
    

    with torch.no_grad():
        out = generator.generate_mesh(data)
        try:
            mesh, stats_dict = out
        except TypeError:
            mesh, stats_dict = out, {}
        print(mesh)
        # vertices = o3d.utility.Vector3dVector(mesh.vertices)
        # triangles = o3d.utility.Vector3iVector(mesh.faces)
        # mesh_o3d = o3d.geometry.TriangleMesh(vertices, triangles)
        path_res = os.path.join('experiments', args.name_exp, 'results', 'airplane')
        if not os.path.exists(path_res):
            os.makedirs(path_res)
        
        path_mesh = os.path.join(path_res, '%s.obj' % 'airplane4')  # 使用数据集名称作为文件名
        mesh.export(path_mesh)
        # print(path_mesh,type(path_mesh))
        # file_path = 'experiments/pretrained_chairs/results/airplane/airplane1.obj'
        # o3d.io.write_triangle_mesh(file_path,mesh_o3d)
