
import torch_geometric.transforms
from torch_geometric.datasets import QM9
from data.transforms import LaplacianPE, SAN_node_LPE, SPE_transform, RRWP_transform, BasisNet_transform

class QM9_loader():
    """Class to load the QM9 dataset in the main files. Applies load_dataset with the transform selected
        from the configuration file"""
    def __init__(self, config):

        self.config = config
        self.location = "datasets/QM9"
        self.transform = None
        self.transform_eval = None
        pass

    def load_data(self, model):


        if model == "SignNet" or (model == "GT" and self.config.GT_dataset.node_encoder_name == "SignNet_PE"):

            self.transform = LaplacianPE(max_freq=self.config.GT_SignNetPE.max_freq)


        elif model == "SAN_node" or (model == "GT" and self.config.GT_dataset.node_encoder_name == "SAN_PE"):
            self.transform = SAN_node_LPE(max_freq=self.config.GT_encLAP.max_freq)


        elif model == "BasisNet" or (model == "GT" and self.config.GT_dataset.node_encoder_name == "BasisNet_PE"):
            self.transform = BasisNet_transform(d=self.config.BasisNet.n_phis)


        elif model == "RWPE" or (model == "GT" and self.config.GT_dataset.node_encoder_name == "RWPE_PE"):
            if model == "GT":
                self.transform = torch_geometric.transforms.AddRandomWalkPE(walk_length=self.config.GT_RWPE.steps)
            else:
                self.transform = torch_geometric.transforms.AddRandomWalkPE(walk_length=self.config.model_RWPE_SAN.pos_enc_dim)

        elif model == "GRIT" or (model == "GT" and self.config.GT_dataset.node_encoder_name == "RRWP_PE"):
            self.transform = RRWP_transform(walk_length=self.config.GT_RWPE.steps )


        elif model == "SPE" or (model == "GT" and self.config.GT_dataset.node_encoder_name == "SPE_PE"):
            self.transform =  SPE_transform(d=self.config.GT_SPE.n_psis)


        else:
            raise ValueError
        # QM9 loader from pytorch geometric using the location and proposed transform
        dataset = QM9(self.location, transform=self.transform)
        mean = dataset.y.mean(dim=0, keepdim=True)
        std = dataset.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.y - mean) / std
        mean, std = mean.to(device="cuda"), std.to(device="cuda")
        train_dataset = dataset[:10000]
        test_dataset = dataset[11000:12000]
        val_dataset = dataset[10000:11000]

        return train_dataset, val_dataset, test_dataset, mean, std

