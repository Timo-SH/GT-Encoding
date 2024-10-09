import torch
import torch_geometric.transforms
from torch_geometric.datasets import ZINC
from data.transforms import LaplacianPE, SAN_node_LPE, SPE_transform, RRWP_transform, BasisNet_transform
import numpy as np
class ZINC_loader():
    """Class to load the ZINC dataset in the main files. Applies load_dataset with the transform selected
    from the configuration file"""
    def __init__(self, config):

        self.config = config
        self.location = "datasets/ZINC"
        self.subset = True
        self.transform = None
        self.transform_eval = None


    def load_data(self, model):



        if model == "SignNet" or (model == "GT" and self.config.GT_dataset.node_encoder_name == "SignNet_PE"):
            self.transform = self.transform_eval = LaplacianPE(max_freq=self.config.GT_SignNetPE.max_freq)

        elif model == "SAN_node" or (model == "GT" and self.config.GT_dataset.node_encoder_name == "SAN_PE"):
            self.transform = self.transform_eval = SAN_node_LPE(max_freq=self.config.GT_encLAP.max_freq)

        elif model == "BasisNet" or (model == "GT" and self.config.GT_dataset.node_encoder_name == "BasisNet_PE"):


            self.transform = BasisNet_transform(d=self.config.BasisNet.n_phis)


        elif model == "RWPE" or (model == "GT" and self.config.GT_dataset.node_encoder_name == "RWPE_PE"):
            if model == "GT":
                self.transform = self.transform_eval = torch_geometric.transforms.AddRandomWalkPE(
                    walk_length=self.config.GT_RWPE.steps)
            else:
                self.transform = self.transform_eval = torch_geometric.transforms.AddRandomWalkPE(walk_length=self.config.model_RWPE_SAN.pos_enc_dim)

        elif model == "GRIT" or (model == "GT" and self.config.GT_dataset.node_encoder_name == "RRWP_PE"):
            self.transform = self.transform_eval = RRWP_transform(walk_length=self.config.GT_RWPE.steps )


        elif model == "SPE" or (model == "GT" and self.config.GT_dataset.node_encoder_name == "SPE_PE"):
            self.transform = self.transform_eval = SPE_transform(d=self.config.GT_SPE.n_psis)


        else:
            pass

        train_dataset = ZINC(self.location, subset=self.subset, split="train", transform=self.transform)
        val_dataset = ZINC(self.location, subset=self.subset, split="val", transform=self.transform_eval)
        test_dataset = ZINC(self.location, subset=self.subset, split="test", transform=self.transform_eval)

        return train_dataset, val_dataset, test_dataset



