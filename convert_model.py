import torch
import numpy as np
from aro_net.Model.aro_export import ARONet

pretrained_model_file_path = './output/4_7080.ckpt'
save_scripted_model_file_path = './output/aronet_cpp.pt'

model = ARONet()

state_dict = torch.load(pretrained_model_file_path, map_location='cpu')["model"]

# FIXME: an extra unused layer values occured
remove_key_list = ["fc_dist_hit.0.weight", "fc_dist_hit.0.bias"]
for remove_key in remove_key_list:
    if remove_key in state_dict.keys():
        del state_dict[remove_key]

model.load_state_dict(state_dict)
model.eval()

pcd = torch.randn(1, 2048, 3)
qry = torch.randn(1, 512, 3)
anc = torch.from_numpy(np.load('./data/anchors/anc_48.npy')).type(torch.float32).unsqueeze(0)
occ = model(pcd, qry, anc)

scripted_model = torch.jit.script(model)

# 保存模型
scripted_model.save(save_scripted_model_file_path)

loaded_model = torch.jit.load(save_scripted_model_file_path)
loaded_occ = loaded_model(pcd, qry, anc)
print(loaded_occ[0, :10])
print(loaded_occ.shape)
print((occ == loaded_occ).all())
