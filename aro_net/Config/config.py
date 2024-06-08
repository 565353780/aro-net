class Config(object):
    def __init__(self) -> None:
        # dataset related
        self.dir_data = "/home/chli/Dataset/aro_net/data"
        self.name_dataset = "shapenet"
        self.name_single = "fertility"
        self.n_wk = 4
        self.categories_train = ["02691156"]
        self.categories_test = ["02691156", "03001627"]
        self.add_noise = 0
        # ARO-Net hyper-parameters
        self.n_pts_train = 2048
        self.n_pts_val = 1024
        self.n_pts_test = 1024
        self.cone_angle_th = 15.0
        self.n_local = 16
        self.n_anc = 48
        self.n_qry = 512
        self.pn_use_bn = False
        self.cond_pn = False
        self.tfm_pos_enc = False
        # common hyper-parameters
        self.device = "cpu"
        self.mode = "test"
        self.n_bs = 1
        self.n_epochs = 600
        self.lr = 1e-5
        self.freq_decay = 100
        self.weight_decay = 0.5
        # Marching Cube realted
        self.mc_chunk_size = 3000
        self.mc_res0 = 16
        self.mc_up_steps = 2
        self.mc_threshold = 0.5

        assert self.name_dataset in ["abc", "shapenet", "single", "custom"]
        assert self.mode in ["train", "test"]
        return


ARO_CONFIG = Config()
