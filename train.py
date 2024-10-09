from config import cfg
import torch
from tqdm import tqdm

from layers.SAN import SAN
from models.GT_model import GT_model


def create_SAN(cfg, edge=False):
    """
        Creates the SAN model from the configuration file.
    """
    model = SAN(cfg, edge_lpe=edge)
    return model



def create_GT(cfg):
    """
    Creates a GT model with the given configuration file
    """
    model = GT_model(cfg)
    return model


def train(train_loader, model, optimizer,loss, device):
    """
    Function to train the model with a defined optimizer using the MAE loss function
    """
    target = 0
    scaler = torch.cuda.amp.GradScaler()
    with tqdm(train_loader, unit="batch") as tepoch:
        total_loss = 0
        N = 0
        for data in tepoch:
            data = data.to(device)
            batch_y = data.y.to(device)
            num_graphs = data.num_graphs
            #Loss and scaler
            loss_calc = (model(data).squeeze() - batch_y).abs().mean()
            scaler.scale(loss_calc).backward()

            #Clipping gradient norm to avoid big gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            total_loss += loss_calc.item() * num_graphs
            #Scaler and optimizer steps
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            N += num_graphs
        return total_loss/N

@torch.no_grad() #disabling gradient calculation during testing
def test(loader, model, device, std=0):
    """
        Computes the test MAE for a given model.
    """
    #Create Error vectors for the Test MAE computation
    if cfg.dataset == "ALCHEMY":
        error = torch.zeros([1,12]).to(device)
        total_error = torch.zeros([1,12]).to(device)
    elif cfg.dataset == "QM9":
        error = torch.zeros([1, 19]).to(device)
        total_error = torch.zeros([1, 19]).to(device)
    else:
        error = 0
    N = 0
    target = 0
    for data in loader:
        #Compute Test MAE for the test data
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y[:,target].to(device), data[0].num_graphs
        else:
            data, y, num_graphs = data.to(device), data.y.to(device), data.num_graphs

        #Compute the explicit test MAE
        if cfg.dataset == "QM9" or cfg.dataset == "ALCHEMY":
            error += ((y * std - model(data) * std).abs() / std).sum(dim=0)
        else:
            error += (model(data).squeeze() - y).abs().sum(dim=0)
    test_perf = error / len(loader.dataset)
    #Either return mean of the loss or loss directly, depending on the dataset
    if cfg.dataset == "QM9" or cfg.dataset =="ALCHEMY":
        return test_perf.mean().item()
    else:
        return test_perf

