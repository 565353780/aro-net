import torch
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
anc = torch.randn(1, 48, 3)
occ = model(pcd, qry, anc)

scripted_model = torch.jit.script(model)

# 保存模型
scripted_model.save(save_scripted_model_file_path)
